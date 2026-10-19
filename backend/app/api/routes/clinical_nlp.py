import json

import logging

import math

from typing import Any



from fastapi import (

    APIRouter,

    Depends,

    File,

    Form,

    HTTPException,

    UploadFile,

)

from sqlalchemy import text

from sqlalchemy.orm import Session



from app.api.dependencies import get_db

from app.auth.dependencies import get_current_user

from app.api.routes.reports import upload_report

from app.api.routes.cases import predict_case

from app.schemas.clinical_entities import CombinedClinicalExtraction

from app.services.clinical_nlp.extraction_service import (

    extract_clinical_entities,

    extract_department_specific_entities,

)

from app.services.chunking.clinical_chunker import chunk_extraction

from app.services.embedding.medcpt_embedding_service import embedding_service



logger = logging.getLogger(__name__)



router = APIRouter(

    prefix="/api/clinical-nlp",

    tags=["Clinical NLP"],

)





def _to_jsonable(value: Any) -> Any:

    """Convert supported model objects into JSON-compatible data."""

    if hasattr(value, "model_dump"):

        return value.model_dump(mode="json")



    if hasattr(value, "dict"):

        return value.dict()



    if isinstance(value, dict):

        return {

            key: _to_jsonable(item)

            for key, item in value.items()

        }



    if isinstance(value, (list, tuple)):

        return [_to_jsonable(item) for item in value]



    return value





def _serialize_chunk(chunk: Any) -> dict:

    """Serialize a chunk without depending on a specific model library."""

    if hasattr(chunk, "model_dump"):

        result = chunk.model_dump(mode="json")

    elif hasattr(chunk, "dict"):

        result = chunk.dict()

    elif isinstance(chunk, dict):

        result = _to_jsonable(chunk)

    else:

        result = {

            "chunk_id": chunk.chunk_id,

            "chunk_type": chunk.chunk_type,

            "text": chunk.text,

            "metadata": _to_jsonable(chunk.metadata),

        }



    if not isinstance(result, dict):

        raise TypeError(

            "Serialized clinical chunk must be a dictionary"

        )



    return result





def _generate_clinical_chunks(

    extraction_json: dict,

    patient_id: int,

    report_id: int,

    case_id: str,

    full_text: str,

) -> dict:

    """Generate parent and child chunks from a clinical extraction."""

    if not isinstance(extraction_json, dict):

        raise ValueError(

            "Clinical extraction must be a JSON object"

        )



    if patient_id <= 0:

        raise ValueError(

            "patient_id must be a positive integer"

        )



    if report_id <= 0:

        raise ValueError(

            "report_id must be a positive integer"

        )



    if not isinstance(case_id, str) or not case_id.strip():

        raise ValueError(

            "case_id must be a non-empty string"

        )
        
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError(
            "full_text must be a non-empty string"
        )



    payload = {

        "patient_id": patient_id,

        "report_id": report_id,

        "case_id": case_id,

        "clinical_extraction": extraction_json,

    }



    result = chunk_extraction(

        payload,

        full_text=full_text,

    )



    if result.parent is None:

        raise ValueError(

            "Clinical chunker did not produce a parent chunk"

        )



    if result.children is None:

        raise ValueError(

            "Clinical chunker did not produce a children collection"

        )



    parent = _serialize_chunk(result.parent)



    children = [

        _serialize_chunk(child)

        for child in result.children

    ]



    chunk_ids = [

        chunk.get("chunk_id")

        for chunk in [parent, *children]

    ]



    if any(chunk_id is None for chunk_id in chunk_ids):

        raise ValueError(

            "A generated chunk is missing its chunk_id"

        )



    if len(chunk_ids) != len(set(chunk_ids)):

        raise ValueError(

            "Clinical chunker generated duplicate chunk IDs"

        )



    return {

        "parent": parent,

        "children": children,

    }





def _generate_chunk_embeddings(

    chunks: dict,

) -> tuple[dict, list]:

    """Generate and validate embeddings while retaining vectors for storage."""

    try:

        parent = chunks.get("parent")

        children = chunks.get("children")



        if not isinstance(parent, dict):

            raise ValueError(

                "Parent chunk must be a dictionary"

            )



        if not isinstance(children, list):

            raise ValueError(

                "Child chunks must be a list"

            )



        all_chunks = [parent, *children]

        texts = []



        for index, chunk in enumerate(all_chunks):

            if not isinstance(chunk, dict):

                raise ValueError(

                    f"Chunk at index {index} must be a dictionary"

                )



            chunk_text = chunk.get("text")



            if (

                not isinstance(chunk_text, str)

                or not chunk_text.strip()

            ):

                raise ValueError(

                    f"Chunk at index {index} has no usable text"

                )



            texts.append(chunk_text)



        vectors = embedding_service.encode_batch(texts)



        if len(vectors) != len(all_chunks):

            raise RuntimeError(

                "Embedding count does not match the chunk count"

            )



        validated_vectors = []



        for index, vector in enumerate(vectors):

            values = [float(value) for value in vector]



            if len(values) != 768:

                raise RuntimeError(

                    f"Chunk at index {index} has an invalid "

                    f"embedding dimension: {len(values)}"

                )



            if not all(math.isfinite(value) for value in values):

                raise RuntimeError(

                    f"Chunk at index {index} has non-finite "

                    "embedding values"

                )



            validated_vectors.append(values)



        logger.info(

            "Clinical chunk embeddings generated successfully",

            extra={

                "chunk_count": len(all_chunks),

                "parent_chunk_count": 1,

                "child_chunk_count": len(children),

                "embedding_dimension": 768,

                "model_name": "ncbi/MedCPT-Article-Encoder",

            },

        )



        summary = {

            "status": "COMPLETED",

            "model": "ncbi/MedCPT-Article-Encoder",

            "embedding_dimension": 768,

            "total_chunks": len(all_chunks),

            "parent_chunks": 1,

            "child_chunks": len(children),

        }



        return summary, validated_vectors



    except Exception:

        logger.exception(

            "Clinical chunk embedding generation failed"

        )

        raise





def _persist_clinical_chunks(

    db: Session,

    chunks: dict,

    vectors: list,

    patient_id: int,

    report_id: int,

    case_id: str,

    extraction_id: int,

) -> int:

    """

    Replace a report's stored chunks within the caller's transaction.



    The caller is responsible for committing or rolling back.

    """

    parent = chunks.get("parent")

    children = chunks.get("children")



    if not isinstance(parent, dict) or not isinstance(children, list):

        raise ValueError(

            "Chunks must contain one parent and a children list"

        )



    all_chunks = [parent, *children]



    if len(vectors) != len(all_chunks):

        raise ValueError(

            "Embedding count does not match the chunk count"

        )



    # Verify the patient, report, case and extraction relationships.

    relationship = db.execute(

        text("""

            SELECT

                r.patient_id,

                c.report_id AS case_report_id,

                ce.report_id AS extraction_report_id,

                ce.case_id AS extraction_case_id

            FROM public.reports AS r

            JOIN public.cases AS c

                ON c.case_id = :case_id

            JOIN public.clinical_extractions AS ce

                ON ce.extraction_id = :extraction_id

            WHERE r.report_id = :report_id

              AND r.patient_id = :patient_id

              AND c.report_id = r.report_id

              AND ce.report_id = r.report_id

              AND ce.case_id = c.case_id

        """),

        {

            "patient_id": patient_id,

            "report_id": report_id,

            "case_id": case_id,

            "extraction_id": extraction_id,

        },

    ).mappings().fetchone()



    if relationship is None:

        raise ValueError(

            "Patient, report, case, and extraction "

            "relationships do not match"

        )



    parent_chunk_id = parent.get("chunk_id")



    if (

        not isinstance(parent_chunk_id, str)

        or not parent_chunk_id.strip()

    ):

        raise ValueError(

            "Parent chunk is missing a valid chunk_id"

        )



    # Validate all chunks before deleting any existing rows.

    rows = []



    for index, chunk in enumerate(all_chunks):

        chunk_id = chunk.get("chunk_id")

        chunk_text = chunk.get("text")

        metadata = chunk.get("metadata") or {}

        chunk_type = chunk.get("chunk_type")



        if not isinstance(chunk_id, str) or not chunk_id.strip():

            raise ValueError(

                f"Chunk at index {index} has an invalid chunk_id"

            )



        if (

            not isinstance(chunk_text, str)

            or not chunk_text.strip()

        ):

            raise ValueError(

                f"Chunk at index {index} has no usable text"

            )



        if not isinstance(metadata, dict):

            raise ValueError(

                f"Chunk at index {index} metadata must be an object"

            )



        metadata_to_store = dict(metadata)



        if chunk_type is not None:

            metadata_to_store.setdefault(

                "chunk_type",

                chunk_type,

            )



        metadata_to_store.setdefault(

            "embedding_model",

            "ncbi/MedCPT-Article-Encoder",

        )



        metadata_to_store.setdefault(

            "embedding_dimension",

            768,

        )



        if index == 0:

            parent_chunk_reference = None

        else:

            metadata_parent_id = metadata_to_store.get("parent_id")



            if metadata_parent_id not in (

                None,

                parent_chunk_id,

            ):

                raise ValueError(

                    f"Child chunk {chunk_id} references "

                    "an unexpected parent ID"

                )



            parent_chunk_reference = parent_chunk_id



        vector_values = vectors[index]



        if len(vector_values) != 768:

            raise ValueError(

                f"Chunk at index {index} must have 768 dimensions"

            )



        if not all(

            math.isfinite(float(value))

            for value in vector_values

        ):

            raise ValueError(

                f"Chunk at index {index} has invalid vector values"

            )



        vector_literal = (

            "["

            + ",".join(

                format(float(value), ".9g")

                for value in vector_values

            )

            + "]"

        )



        rows.append({

            "chunk_id": chunk_id,

            "parent_chunk_id": parent_chunk_reference,

            "patient_id": patient_id,

            "report_id": report_id,

            "case_id": case_id,

            "extraction_id": extraction_id,

            "chunk_text": chunk_text,

            "metadata": json.dumps(

                metadata_to_store,

                ensure_ascii=False,

            ),

            "embedding": vector_literal,

        })



    # Remove old children before the parent because of the self-FK.

    # These changes are rolled back if the transaction fails.

    db.execute(

        text("""

            DELETE FROM public.clinical_chunks

            WHERE report_id = :report_id

              AND parent_chunk_id IS NOT NULL

        """),

        {"report_id": report_id},

    )



    db.execute(

        text("""

            DELETE FROM public.clinical_chunks

            WHERE report_id = :report_id

              AND parent_chunk_id IS NULL

        """),

        {"report_id": report_id},

    )



    # Parent is first in rows, followed by the child chunks.

    for row in rows:

        db.execute(

            text("""

                INSERT INTO public.clinical_chunks (

                    chunk_id,

                    parent_chunk_id,

                    patient_id,

                    report_id,

                    case_id,

                    extraction_id,

                    chunk_text,

                    metadata,

                    embedding

                )

                VALUES (

                    :chunk_id,

                    :parent_chunk_id,

                    :patient_id,

                    :report_id,

                    :case_id,

                    :extraction_id,

                    :chunk_text,

                    CAST(:metadata AS JSONB),

                    CAST(:embedding AS vector)

                )

            """),

            row,

        )



    logger.info(

        "Clinical chunks and embeddings persisted",

        extra={

            "patient_id": patient_id,

            "report_id": report_id,

            "case_id": case_id,

            "extraction_id": extraction_id,

            "chunk_count": len(rows),

        },

    )



    return len(rows)





@router.post("/process")

async def process_clinical_report(

    patient_id: int = Form(...),

    file: UploadFile = File(...),

    db: Session = Depends(get_db),

    current_user: dict = Depends(get_current_user),

):

    """

    Upload a report, predict its department, extract clinical entities,

    generate chunks and embeddings, and persist them in PostgreSQL.

    """

    try:

        # Step 1: Validate the request.

        if patient_id <= 0:

            raise HTTPException(

                status_code=422,

                detail="patient_id must be a positive integer",

            )



        if not file.filename:

            raise HTTPException(

                status_code=422,

                detail="A report filename is required",

            )



        user_id = current_user.get("user_id")



        if user_id is None:

            raise HTTPException(

                status_code=401,

                detail="Authenticated user ID is missing",

            )



        # Step 2: Reuse the existing report upload pipeline.

        upload_result = await upload_report(

            patient_id=patient_id,

            file=file,

            db=db,

            current_user=current_user,

        )



        report_id = upload_result.get("report_id")

        case_id = upload_result.get("case_id")



        if report_id is None or not case_id:

            raise HTTPException(

                status_code=500,

                detail="Upload did not return report_id and case_id",

            )



        try:

            report_id = int(report_id)

        except (TypeError, ValueError) as exc:

            raise HTTPException(

                status_code=500,

                detail="Upload returned an invalid report_id",

            ) from exc



        if report_id <= 0:

            raise HTTPException(

                status_code=500,

                detail="Upload returned an invalid report_id",

            )



        # Step 3: Check for an existing extraction.

        existing = db.execute(

            text("""

                SELECT

                    extraction_id,

                    report_id,

                    case_id,

                    prediction_id,

                    predicted_department,

                    extraction_json,

                    extraction_status,

                    created_at,

                    updated_at

                FROM public.clinical_extractions

                WHERE report_id = :report_id

            """),

            {"report_id": report_id},

        ).mappings().fetchone()



        if existing is not None:

            existing_record = dict(existing)



            if existing_record["case_id"] != case_id:

                raise HTTPException(

                    status_code=409,

                    detail=(

                        "The stored extraction case does not "

                        "match the uploaded case"

                    ),

                )



            if existing_record["extraction_status"] != "COMPLETED":

                raise HTTPException(

                    status_code=409,

                    detail="The existing clinical extraction is not completed",

                )



            existing_extraction_json = existing_record["extraction_json"]



            if isinstance(existing_extraction_json, str):

                try:

                    existing_extraction_json = json.loads(

                        existing_extraction_json

                    )

                except json.JSONDecodeError as exc:

                    logger.exception(

                        "Stored extraction JSON is invalid "

                        "for report_id=%s",

                        report_id,

                    )

                    raise HTTPException(

                        status_code=500,

                        detail=(

                            "Stored clinical extraction "

                            "contains invalid JSON"

                        ),

                    ) from exc



            try:

                report_text_result = db.execute(
                    text("""
                        SELECT extracted_text
                        FROM public.reports
                        WHERE report_id = :report_id
                          AND patient_id = :patient_id
                    """),
                    {
                        "report_id": report_id,
                        "patient_id": patient_id,
                    },
                ).mappings().fetchone()

                if (
                    report_text_result is None
                    or not isinstance(report_text_result["extracted_text"], str)
                    or not report_text_result["extracted_text"].strip()
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="Original report text is unavailable for clinical chunking",
                    )

                existing_report_text = report_text_result["extracted_text"]

                chunks = _generate_clinical_chunks(
                    extraction_json=existing_extraction_json,
                    patient_id=patient_id,
                    report_id=report_id,
                    case_id=case_id,
                    full_text=existing_report_text,
                )



                embedding_summary, embedding_vectors = (

                    _generate_chunk_embeddings(chunks)

                )



                stored_chunk_count = _persist_clinical_chunks(

                    db=db,

                    chunks=chunks,

                    vectors=embedding_vectors,

                    patient_id=patient_id,

                    report_id=report_id,

                    case_id=case_id,

                    extraction_id=int(

                        existing_record["extraction_id"]

                    ),

                )



                db.commit()



                embedding_summary["persisted"] = True

                embedding_summary["stored_chunks"] = stored_chunk_count



            except Exception as exc:

                db.rollback()



                logger.exception(

                    "Chunk processing or persistence failed "

                    "for existing report_id=%s, patient_id=%s",

                    report_id,

                    patient_id,

                )



                if isinstance(exc, HTTPException):

                    raise



                raise HTTPException(

                    status_code=500,

                    detail="Clinical chunk processing failed",

                ) from exc



            return {

                "message": "Clinical extraction already exists",

                "patient_id": patient_id,

                "report_id": report_id,

                "case_id": case_id,

                "extraction": existing_record,

                "chunks": chunks,

                "embedding_summary": embedding_summary,

            }



        # Step 4: Generate or retrieve the SVM prediction.

        case_result = db.execute(

            text("""

                SELECT status

                FROM public.cases

                WHERE case_id = :case_id

            """),

            {"case_id": case_id},

        ).mappings().fetchone()



        if case_result is None:

            raise HTTPException(

                status_code=404,

                detail="The uploaded case could not be found",

            )



        if case_result["status"] == "Pending Prediction":

            prediction = predict_case(

                case_id=case_id,

                db=db,

            )



        elif case_result["status"] == "Predicted":

            prediction_result = db.execute(

                text("""

                    SELECT

                        p.prediction_id,

                        p.case_id,

                        p.predicted_department_id,

                        d.department_name AS predicted_department,

                        p.confidence_score,

                        p.confidence_level_label,

                        p.decision_function_scores,

                        p.model_version

                    FROM public.predictions AS p

                    JOIN public.departments AS d

                        ON d.department_id = p.predicted_department_id

                    WHERE p.case_id = :case_id

                    ORDER BY p.prediction_id DESC

                    LIMIT 1

                """),

                {"case_id": case_id},

            ).mappings().fetchone()



            if prediction_result is None:

                raise HTTPException(

                    status_code=409,

                    detail=(

                        "The case is marked Predicted, "

                        "but no prediction record exists"

                    ),

                )



            prediction = dict(prediction_result)



        else:

            raise HTTPException(

                status_code=409,

                detail=(

                    "The case cannot be predicted in its current state: "

                    f"{case_result['status']}"

                ),

            )



        prediction_id = prediction.get("prediction_id")

        predicted_department_id = prediction.get(

            "predicted_department_id"

        )



        if (

            prediction_id is None

            or predicted_department_id is None

        ):

            raise HTTPException(

                status_code=500,

                detail="Prediction is missing required identifiers",

            )



        # Step 5: Read the extracted report text.

        report_result = db.execute(

            text("""

                SELECT extracted_text

                FROM public.reports

                WHERE report_id = :report_id

            """),

            {"report_id": report_id},

        ).mappings().fetchone()



        if report_result is None:

            raise HTTPException(

                status_code=404,

                detail="The uploaded report could not be found",

            )



        report_text = report_result["extracted_text"]



        if (

            not isinstance(report_text, str)

            or not report_text.strip()

        ):

            raise HTTPException(

                status_code=422,

                detail="The report contains no usable extracted text",

            )



        # Step 6: Extract common and department-specific entities.

        common_result = extract_clinical_entities(report_text)



        department_specific_result = (

            extract_department_specific_entities(report_text)

        )



        common_json = _to_jsonable(common_result)



        department_specific_json = _to_jsonable(

            department_specific_result

        )



        if not isinstance(common_json, dict):

            raise TypeError(

                "Common clinical extractor must return "

                "a model or dictionary"

            )



        if not isinstance(department_specific_json, dict):

            raise TypeError(

                "Department-specific extractor must return a dictionary"

            )



        combined_result = CombinedClinicalExtraction(

            patient_id=patient_id,

            common=common_json,

            department_specific=department_specific_json,

        )



        extraction_json = _to_jsonable(combined_result)



        if not isinstance(extraction_json, dict):

            raise TypeError(

                "Combined clinical extraction must be a JSON object"

            )



        # Step 7: Generate chunks and embeddings.

        try:

            chunks = _generate_clinical_chunks(
                extraction_json=extraction_json,
                patient_id=patient_id,
                report_id=report_id,
                case_id=case_id,
                full_text=report_text,
            )



            embedding_summary, embedding_vectors = (

                _generate_chunk_embeddings(chunks)

            )



        except Exception as exc:

            logger.exception(

                "Clinical chunking or embedding failed "

                "for report_id=%s, patient_id=%s, case_id=%s",

                report_id,

                patient_id,

                case_id,

            )



            raise HTTPException(

                status_code=500,

                detail="Clinical chunk generation failed",

            ) from exc



        # Step 8: Save the extraction.

        saved_result = db.execute(

            text("""

                INSERT INTO public.clinical_extractions (

                    report_id,

                    case_id,

                    prediction_id,

                    predicted_department,

                    extraction_json,

                    extraction_status,

                    extracted_by

                )

                VALUES (

                    :report_id,

                    :case_id,

                    :prediction_id,

                    :predicted_department_id,

                    CAST(:extraction_json AS JSONB),

                    'COMPLETED',

                    :extracted_by

                )

                RETURNING

                    extraction_id,

                    report_id,

                    case_id,

                    prediction_id,

                    predicted_department,

                    extraction_json,

                    extraction_status,

                    created_at,

                    updated_at

            """),

            {

                "report_id": report_id,

                "case_id": case_id,

                "prediction_id": prediction_id,

                "predicted_department_id": predicted_department_id,

                "extraction_json": json.dumps(

                    extraction_json,

                    ensure_ascii=False,

                ),

                "extracted_by": user_id,

            },

        ).mappings().fetchone()



        if saved_result is None:

            raise RuntimeError(

                "Clinical extraction insert returned no saved record"

            )



        # Step 9: Persist chunks and embeddings in the same transaction.

        stored_chunk_count = _persist_clinical_chunks(

            db=db,

            chunks=chunks,

            vectors=embedding_vectors,

            patient_id=patient_id,

            report_id=report_id,

            case_id=case_id,

            extraction_id=int(saved_result["extraction_id"]),

        )



        db.commit()



        embedding_summary["persisted"] = True

        embedding_summary["stored_chunks"] = stored_chunk_count



        # Step 10: Return the extraction, chunks and embedding summary.

        return {

            "message": "Report processed successfully",

            "patient_id": patient_id,

            "report_id": report_id,

            "case_id": case_id,

            "prediction": prediction,

            "clinical_extraction": dict(saved_result),

            "chunks": chunks,

            "embedding_summary": embedding_summary,

        }



    except HTTPException:

        db.rollback()

        raise



    except Exception as exc:

        db.rollback()



        logger.exception(

            "Clinical NLP processing failed "

            "for patient_id=%s, filename=%s",

            patient_id,

            file.filename,

        )



        raise HTTPException(

            status_code=500,

            detail=(

                "Clinical NLP processing failed. "

                "Check the backend logs."

            ),

        ) from exc





@router.get("/reports/{report_id}")

def get_clinical_extraction(

    report_id: int,

    db: Session = Depends(get_db),

    current_user: dict = Depends(get_current_user),

):

    """Retrieve the saved clinical extraction for a report."""

    try:

        result = db.execute(

            text("""

                SELECT

                    extraction_id,

                    report_id,

                    case_id,

                    prediction_id,

                    predicted_department,

                    extraction_json,

                    extraction_status,

                    extraction_error,

                    extracted_by,

                    created_at,

                    updated_at

                FROM public.clinical_extractions

                WHERE report_id = :report_id

            """),

            {"report_id": report_id},

        ).mappings().fetchone()



        if result is None:

            raise HTTPException(

                status_code=404,

                detail="Clinical extraction not found for this report",

            )



        return dict(result)



    except HTTPException:

        raise



    except Exception as exc:

        logger.exception(

            "Failed to retrieve clinical extraction "

            "for report_id=%s",

            report_id,

        )



        raise HTTPException(

            status_code=500,

            detail="Failed to retrieve clinical extraction",

        ) from exc
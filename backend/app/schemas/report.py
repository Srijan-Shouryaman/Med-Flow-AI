from datetime import datetime
from pydantic import BaseModel


class ReportResponse(BaseModel):
    report_id: int
    patient_id: int
    original_filename: str
    stored_filename: str
    file_path: str
    content_type: str
    file_size: int
    sha256_hash: str
    extracted_text: str | None
    extraction_status: str
    extraction_error: str | None
    uploaded_by: int
    uploaded_at: datetime
    case_id: str | None = None
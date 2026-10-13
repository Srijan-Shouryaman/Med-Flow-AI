import { useEffect, useMemo, useState } from "react";
import { ArrowRight, Search, UserRoundPlus } from "lucide-react";
import { Link } from "react-router";
import Card from "../components/ui/Card.jsx";
import FeedbackState from "../components/ui/FeedbackState.jsx";
import PageHeader from "../components/ui/PageHeader.jsx";
import { formatDate, loadFailureKind } from "../utils/formatters.js";
import { getPatients } from "../services/patientApi.js";

export default function PatientsPage() {
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState({ status: "loading", patients: [] });
  const [search, setSearch] = useState("");

  useEffect(() => {
    let active = true;
    setState({ status: "loading", patients: [] });
    getPatients()
      .then((patients) => {
        if (active) setState({ status: "success", patients });
      })
      .catch((error) => {
        if (active) setState({ status: loadFailureKind(error), patients: [] });
      });
    return () => { active = false; };
  }, [revision]);

  const filteredPatients = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    if (!query) return state.patients;
    return state.patients.filter((patient) => [
      patient.patient_ref_id,
      patient.full_name,
      patient.gender,
      patient.phone,
      patient.email,
    ].some((value) => String(value ?? "").toLocaleLowerCase().includes(query)));
  }, [search, state.patients]);

  return (
    <>
      <PageHeader title="Patients" />
      <Card className="patient-create-card">
        <span className="patient-create-icon"><UserRoundPlus size={19} aria-hidden="true" /></span>
        <div className="patient-create-copy">
          <h2>Add a new patient</h2>
          <p>Create a patient record to manage medical reports and cases.</p>
        </div>
        <Link className="patient-primary-button patient-create-link" to="/patients/new">
          Add patient <ArrowRight size={15} aria-hidden="true" />
        </Link>
      </Card>

      <Card className="patient-list-card">
        <div className="patient-list-toolbar">
          <div>
            <p className="card-eyebrow">Patient directory</p>
            <h2>Patient records</h2>
          </div>
          <label className="patient-search">
            <Search size={16} aria-hidden="true" />
            <span className="sr-only">Search patients</span>
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by name, reference, or contact"
            />
          </label>
        </div>

        {state.status === "loading" ? (
          <FeedbackState type="loading" title="Loading patient records..." />
        ) : state.status !== "success" ? (
          <div className="patient-load-error">
            <FeedbackState
              type={state.status}
              title={state.status === "unauthorized" ? "Patient records are unavailable for this account." : state.status === "network" ? "Could not connect to patient records." : "Unable to load patient records."}
              description="Please try again. If the issue continues, contact your administrator."
            />
            <button className="patient-secondary-button" type="button" onClick={() => setRevision((value) => value + 1)}>Retry</button>
          </div>
        ) : filteredPatients.length === 0 ? (
          <FeedbackState
            type="empty"
            title={search.trim() ? "No matching patients found." : "No patient records yet."}
            description={search.trim() ? "Try another name, reference ID, or contact detail." : "Add a patient record to begin report intake."}
          />
        ) : (
          <div className="patient-table-scroll">
            <table className="patient-table">
              <thead>
                <tr>
                  <th scope="col">Patient reference</th>
                  <th scope="col">Patient</th>
                  <th scope="col">Date of birth</th>
                  <th scope="col">Gender</th>
                  <th scope="col">Phone</th>
                  <th scope="col">Email</th>
                  <th scope="col">Created</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map((patient) => (
                  <tr key={patient.patient_id}>
                    <td><span className="patient-reference">{patient.patient_ref_id || "—"}</span></td>
                    <td><strong className="patient-name">{patient.full_name || "—"}</strong></td>
                    <td>{formatDate(patient.date_of_birth)}</td>
                    <td>{patient.gender || "—"}</td>
                    <td>{patient.phone || "—"}</td>
                    <td>{patient.email || "—"}</td>
                    <td>{formatDate(patient.created_at, { includeTime: true })}</td>
                    <td>
                      <Link className="patient-view-link" to={`/patients/${patient.patient_id}`}>
                        View patient <ArrowRight size={14} aria-hidden="true" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {state.status === "success" && state.patients.length > 0 ? (
          <p className="patient-table-footnote">Showing {filteredPatients.length} of {state.patients.length} patient records</p>
        ) : null}
      </Card>
    </>
  );
}

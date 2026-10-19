import { useEffect, useMemo, useState } from "react";
import { ArrowRight, FileText, FileUp, Search } from "lucide-react";
import { Link } from "react-router";
import Card from "../components/ui/Card.jsx";
import FeedbackState from "../components/ui/FeedbackState.jsx";
import StatusBadge from "../components/ui/StatusBadge.jsx";
import { formatDate, formatFileSize, loadFailureKind } from "../utils/formatters.js";
import { getReports } from "../services/workflowApi.js";

const extractionFilters = ["All", "EXTRACTED", "FAILED", "PENDING"];

export default function ReportsPage() {
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState({ status: "loading", reports: [] });
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");

  useEffect(() => {
    let active = true;
    setState({ status: "loading", reports: [] });
    getReports()
      .then((reports) => { if (active) setState({ status: "success", reports }); })
      .catch((error) => { if (active) setState({ status: loadFailureKind(error), reports: [] }); });
    return () => { active = false; };
  }, [revision]);

  const filteredReports = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return state.reports.filter((report) => {
      const searchable = [report.original_filename, report.patient_ref_id, report.patient_name, report.case_id]
        .some((value) => String(value ?? "").toLocaleLowerCase().includes(query));
      const statusMatches = statusFilter === "All"
        || String(report.extraction_status ?? "").toUpperCase() === statusFilter;
      return searchable && statusMatches;
    });
  }, [search, state.reports, statusFilter]);

  return (
    <>
      <Card className="workflow-list-card">
        <div className="workflow-list-toolbar">
          <div>
            <p className="card-eyebrow">Report history</p>
            <h2>Uploaded reports</h2>
          </div>
          <div className="workflow-list-controls">
            <Link className="patient-primary-button report-upload-link" to="/reports/upload">
              <FileUp size={15} aria-hidden="true" /> Upload report
            </Link>
            <label className="patient-search">
              <Search size={16} aria-hidden="true" />
              <span className="sr-only">Search reports</span>
              <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search filename, patient, or case" />
            </label>
            <label className="workflow-filter">
              <span>Extraction</span>
              <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} aria-label="Filter by extraction status">
                {extractionFilters.map((status) => <option key={status} value={status}>{status === "All" ? "All statuses" : status}</option>)}
              </select>
            </label>
          </div>
        </div>

        {state.status === "loading" ? (
          <FeedbackState type="loading" title="Loading report history..." />
        ) : state.status !== "success" ? (
          <div className="patient-load-error">
            <FeedbackState
              type={state.status}
              title={state.status === "unauthorized" ? "Report history is unavailable for this account." : state.status === "network" ? "Could not connect to report history." : "Unable to load report history."}
              description="Please try again. If the issue continues, contact your administrator."
            />
            <button className="patient-secondary-button" type="button" onClick={() => setRevision((value) => value + 1)}>Retry</button>
          </div>
        ) : filteredReports.length === 0 ? (
          <FeedbackState
            type="empty"
            title={state.reports.length ? "No matching reports found." : "No reports uploaded yet."}
            description={state.reports.length ? "Change the search or extraction filter." : "Reports uploaded from a patient profile will appear here."}
          />
        ) : (
          <div className="patient-table-scroll">
            <table className="patient-table report-table">
              <thead>
                <tr>
                  <th scope="col">Report</th>
                  <th scope="col">Patient</th>
                  <th scope="col">Case</th>
                  <th scope="col">Extraction</th>
                  <th scope="col">File type</th>
                  <th scope="col">Size</th>
                  <th scope="col">Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {filteredReports.map((report) => (
                  <tr key={report.report_id}>
                    <td>
                      <Link className="workflow-record-link" to={`/reports/${report.report_id}`}>
                        <FileText size={15} aria-hidden="true" />
                        <span><strong>{report.original_filename || `Report #${report.report_id}`}</strong><small>Report #{report.report_id}</small></span>
                      </Link>
                    </td>
                    <td>
                      {report.patient_id != null ? (
                        <Link className="workflow-patient-link" to={`/patients/${report.patient_id}`}>
                          <strong>{report.patient_name || "Patient profile"}</strong><small>{report.patient_ref_id || "Reference unavailable"}</small>
                        </Link>
                      ) : <span>{report.patient_name || report.patient_ref_id || "—"}</span>}
                    </td>
                    <td>{report.case_id
                      ? <Link className="patient-reference" to={`/cases/${encodeURIComponent(report.case_id)}`}>{report.case_id} <ArrowRight size={12} aria-hidden="true" /></Link>
                      : <span className="table-muted">—</span>}</td>
                    <td><StatusBadge status={String(report.extraction_status ?? "Unknown").toLowerCase()}>{report.extraction_status || "Unknown"}</StatusBadge></td>
                    <td>{report.content_type || "—"}</td>
                    <td>{formatFileSize(report.file_size)}</td>
                    <td>{formatDate(report.uploaded_at, { includeTime: true })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {state.status === "success" && state.reports.length ? <p className="patient-table-footnote">Showing {filteredReports.length} of {state.reports.length} reports</p> : null}
      </Card>
    </>
  );
}

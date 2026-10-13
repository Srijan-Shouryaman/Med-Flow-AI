import { BrowserRouter, Route, Routes } from "react-router";
import { useNavigate } from "react-router";
import { AuthProvider, useAuth } from "./auth/AuthContext.jsx";
import AppShell from "./components/layout/AppShell.jsx";
import ProtectedRoute from "./components/routing/ProtectedRoute.jsx";
import { navigationSections } from "./config/navigation.js";
import DashboardPage from "./pages/DashboardPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import ModulePlaceholderPage from "./pages/ModulePlaceholderPage.jsx";
import NotFoundPage from "./pages/NotFoundPage.jsx";
import TeamPerformancePage from "./pages/TeamPerformancePage.jsx";
import PatientsPage from "./pages/PatientsPage.jsx";
import PatientDetailPage from "./pages/PatientDetailPage.jsx";
import PatientFormPage from "./pages/PatientFormPage.jsx";
import ReportUploadPage from "./pages/ReportUploadPage.jsx";
import CasesPage from "./pages/CasesPage.jsx";
import CaseDetailPage from "./pages/CaseDetailPage.jsx";
import ReportsPage from "./pages/ReportsPage.jsx";
import ReportDetailPage from "./pages/ReportDetailPage.jsx";
import PredictionsPage from "./pages/PredictionsPage.jsx";
import RecommendationsPage from "./pages/RecommendationsPage.jsx";
import AssignmentsPage from "./pages/AssignmentsPage.jsx";
import AssignmentDetailPage from "./pages/AssignmentDetailPage.jsx";
import UpdateCaseStatusPage from "./pages/UpdateCaseStatusPage.jsx";
import PredictionDetailPage from "./pages/PredictionDetailPage.jsx";
import DiagnosisPage from "./pages/DiagnosisPage.jsx";

const modulePages = navigationSections.flatMap((section) =>
  section.items
    .filter((item) => item.href !== "/")
    .map((item) => ({ ...item, section: section.label })),
);

function AuthenticatedAppShell() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate("/login", { replace: true });
  }

  return (
    <AppShell user={user} onLogout={handleLogout} />
  );
}

function AppRoutes() {
  return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<AuthenticatedAppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="patients" element={<PatientsPage />} />
            <Route path="patients/new" element={<PatientFormPage />} />
            <Route path="patients/:patientId/upload" element={<ReportUploadPage />} />
            <Route path="patients/:patientId" element={<PatientDetailPage />} />
            <Route path="reports" element={<ReportsPage />} />
            <Route path="reports/:reportId" element={<ReportDetailPage />} />
            <Route path="cases" element={<CasesPage />} />
            <Route path="cases/:caseId" element={<CaseDetailPage />} />
            <Route path="predictions" element={<PredictionsPage />} />
            <Route path="predictions/:predictionId" element={<PredictionDetailPage />} />
            <Route path="recommendations" element={<RecommendationsPage />} />
            <Route path="assignments" element={<AssignmentsPage />} />
            <Route path="assignments/:assignmentId" element={<AssignmentDetailPage />} />
            <Route path="case-status" element={<UpdateCaseStatusPage />} />
            {modulePages.map((module) => (
              ["/patients", "/reports", "/cases", "/predictions", "/recommendations", "/assignments", "/case-status"].includes(module.href) ? null :
              <Route
                key={module.href}
                path={module.href.slice(1)}
                element={module.href === "/team-performance"
                  ? <TeamPerformancePage />
                  : module.href === "/diagnosis"
                    ? <DiagnosisPage />
                  : <ModulePlaceholderPage module={module} />}
              />
            ))}
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Route>
      </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}

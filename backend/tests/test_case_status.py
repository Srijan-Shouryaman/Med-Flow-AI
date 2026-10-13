import json
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.api.routes.cases import router as cases_router
from app.auth.dependencies import get_current_user


class FakeResult:
    def __init__(self, row=None, rowcount=0):
        self.row = row
        self.rowcount = rowcount

    def mappings(self):
        return self

    def fetchone(self):
        return self.row


class FakeDatabase:
    def __init__(self, status="Assigned", has_assignment=True):
        self.status = status
        self.has_assignment = has_assignment
        self.audit_logs = []
        self._committed_status = status
        self._committed_audit_logs = []
        self.statements = []
        self.force_update_miss = False
        self.fail_audit_insert = False

    def execute(self, statement, params=None):
        sql = str(statement)
        normalized = " ".join(sql.lower().split())
        params = params or {}
        self.statements.append((normalized, params))

        if "from cases" in normalized and normalized.startswith("select"):
            if "case_id" not in params or params["case_id"] == "missing":
                return FakeResult()
            return FakeResult({"case_id": params["case_id"], "status": self.status})

        if "from assignments" in normalized and normalized.startswith("select"):
            row = {"assignment_id": 17} if self.has_assignment else None
            return FakeResult(row)

        if normalized.startswith("update cases"):
            if self.force_update_miss or self.status != params["current_status"]:
                return FakeResult(rowcount=0)
            self.status = params["new_status"]
            return FakeResult(rowcount=1)

        if normalized.startswith("insert into audit_logs"):
            if self.fail_audit_insert:
                raise RuntimeError("simulated audit storage failure")
            self.audit_logs.append(params.copy())
            return FakeResult(rowcount=1)

        raise AssertionError(f"Unexpected SQL in test: {sql}")

    def commit(self):
        self._committed_status = self.status
        self._committed_audit_logs = list(self.audit_logs)

    def rollback(self):
        self.status = self._committed_status
        self.audit_logs = list(self._committed_audit_logs)


class CaseStatusEndpointTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDatabase()
        self.user = {
            "full_name": "Test Clinician",
            "email": "clinician@example.test",
            "role": "SUPER_ADMIN",
        }
        app = FastAPI()
        app.include_router(cases_router)

        def provide_db():
            yield self.db

        app.dependency_overrides[get_db] = provide_db
        app.dependency_overrides[get_current_user] = lambda: self.user
        self.app = app
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def patch_status(self, case_id="C0102", status="In Progress"):
        return self.client.patch(
            f"/api/cases/{case_id}/status",
            json={"status": status},
        )

    def test_assigned_case_with_final_assignment_moves_to_in_progress_and_is_audited(self):
        response = self.patch_status()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "case_id": "C0102",
            "previous_status": "Assigned",
            "status": "In Progress",
            "actor": "Test Clinician",
        })
        self.assertEqual(self.db.status, "In Progress")
        self.assertEqual(len(self.db.audit_logs), 1)
        self.assertEqual(self.db.audit_logs[0]["event_type"], "CASE_STATUS_CHANGED")
        self.assertEqual(self.db.audit_logs[0]["actor"], "Test Clinician")
        self.assertEqual(json.loads(self.db.audit_logs[0]["details"]), {
            "previous_status": "Assigned",
            "new_status": "In Progress",
        })
        self.assertTrue(any(
            "final_assigned_team_id is not null" in sql
            for sql, _params in self.db.statements
        ))
        self.assertIn("for update", self.db.statements[0][0])

    def test_in_progress_case_moves_to_resolved(self):
        self.db.status = "In Progress"
        self.db._committed_status = "In Progress"

        response = self.patch_status(status="Resolved")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "Resolved")
        self.assertEqual(self.db.status, "Resolved")
        self.assertEqual(len(self.db.audit_logs), 1)

    def test_allowed_roles_can_update_status(self):
        for role in ("SUPER_ADMIN", "HOSPITAL_ADMIN", "DEPARTMENT_HEAD", "DOCTOR"):
            with self.subTest(role=role):
                self.db = FakeDatabase()
                self.user = {"full_name": role, "email": "staff@example.test", "role": role}
                response = self.patch_status()
                self.assertEqual(response.status_code, 200)

    def test_nurse_is_forbidden(self):
        self.user = {"full_name": "Nurse", "email": "nurse@example.test", "role": "NURSE"}

        response = self.patch_status()

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.db.status, "Assigned")
        self.assertEqual(self.db.audit_logs, [])

    def test_missing_case_returns_404(self):
        response = self.patch_status(case_id="missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.db.audit_logs, [])

    def test_missing_final_assignment_rejects_in_progress_transition(self):
        self.db.has_assignment = False

        response = self.patch_status()

        self.assertEqual(response.status_code, 409)
        self.assertIn("assignment is required", response.json()["detail"])
        self.assertEqual(self.db.status, "Assigned")
        self.assertEqual(self.db.audit_logs, [])

    def test_unsupported_and_stale_transitions_are_rejected_without_duplicate_audit(self):
        self.db.status = "Pending Prediction"
        self.db._committed_status = "Pending Prediction"
        invalid = self.patch_status()
        self.assertEqual(invalid.status_code, 409)
        self.assertEqual(self.db.audit_logs, [])

        self.db = FakeDatabase()
        first = self.patch_status()
        second = self.patch_status()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(self.db.status, "In Progress")
        self.assertEqual(len(self.db.audit_logs), 1)

    def test_update_miss_returns_conflict_without_audit(self):
        self.db.force_update_miss = True

        response = self.patch_status()

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.db.status, "Assigned")
        self.assertEqual(self.db.audit_logs, [])

    def test_audit_failure_rolls_back_status_change(self):
        self.db.fail_audit_insert = True

        response = self.patch_status()

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Failed to update case status")
        self.assertEqual(self.db.status, "Assigned")
        self.assertEqual(self.db.audit_logs, [])

    def test_missing_bearer_token_is_rejected(self):
        self.app.dependency_overrides.pop(get_current_user)

        response = self.patch_status()

        self.assertIn(response.status_code, (401, 403))
        self.assertEqual(self.db.status, "Assigned")
        self.assertEqual(self.db.audit_logs, [])


if __name__ == "__main__":
    unittest.main()

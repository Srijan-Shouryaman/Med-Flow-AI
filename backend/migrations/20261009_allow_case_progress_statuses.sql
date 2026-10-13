BEGIN;

ALTER TABLE cases
    DROP CONSTRAINT IF EXISTS cases_status_check;

ALTER TABLE cases
    ADD CONSTRAINT cases_status_check
    CHECK (
        status IN (
            'Pending Prediction',
            'Predicted',
            'Recommended',
            'Pending Human Review',
            'Assigned',
            'Flagged - No Eligible Team',
            'In Progress',
            'Resolved'
        )
    );

COMMIT;

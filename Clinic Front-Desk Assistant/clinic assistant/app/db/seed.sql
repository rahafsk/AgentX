-- Fictional local-development fixtures only. Never use these as real clinic data.
-- initialize.py sets app.environment only when APP_ENV=development; this SQL
-- guard also prevents accidental direct execution in other environments.
DO $demo_seed_guard$
BEGIN
    IF current_setting('app.environment', true) IS DISTINCT FROM 'development' THEN
        RAISE EXCEPTION
            'Demo seed data is development-only; set APP_ENV=development to load it.';
    END IF;
END;
$demo_seed_guard$;

INSERT INTO doctors (id, full_name, specialty, is_active)
VALUES
    (
        '00000000-0000-4000-8000-000000000001',
        'Demo Provider Alpha (fictional)',
        'DEMO ONLY',
        TRUE
    ),
    (
        '00000000-0000-4000-8000-000000000002',
        'Demo Provider Beta (fictional)',
        'DEMO ONLY',
        TRUE
    )
ON CONFLICT (id) DO NOTHING;

WITH demo_slots (id, doctor_id, starts_at, ends_at) AS (
    SELECT
        '00000000-0000-4000-8000-000000000101'::UUID,
        '00000000-0000-4000-8000-000000000001'::UUID,
        (
            date_trunc('day', timezone('UTC', CURRENT_TIMESTAMP))
            + INTERVAL '1 day 10 hours'
        ) AT TIME ZONE 'UTC',
        (
            date_trunc('day', timezone('UTC', CURRENT_TIMESTAMP))
            + INTERVAL '1 day 10 hours 30 minutes'
        ) AT TIME ZONE 'UTC'
    UNION ALL
    SELECT
        '00000000-0000-4000-8000-000000000102'::UUID,
        '00000000-0000-4000-8000-000000000001'::UUID,
        (
            date_trunc('day', timezone('UTC', CURRENT_TIMESTAMP))
            + INTERVAL '1 day 11 hours'
        ) AT TIME ZONE 'UTC',
        (
            date_trunc('day', timezone('UTC', CURRENT_TIMESTAMP))
            + INTERVAL '1 day 11 hours 30 minutes'
        ) AT TIME ZONE 'UTC'
    UNION ALL
    SELECT
        '00000000-0000-4000-8000-000000000103'::UUID,
        '00000000-0000-4000-8000-000000000002'::UUID,
        (
            date_trunc('day', timezone('UTC', CURRENT_TIMESTAMP))
            + INTERVAL '1 day 14 hours'
        ) AT TIME ZONE 'UTC',
        (
            date_trunc('day', timezone('UTC', CURRENT_TIMESTAMP))
            + INTERVAL '1 day 14 hours 30 minutes'
        ) AT TIME ZONE 'UTC'
)
INSERT INTO appointment_slots (id, doctor_id, starts_at, ends_at, status)
SELECT id, doctor_id, starts_at, ends_at, 'open'
FROM demo_slots
ON CONFLICT (id) DO UPDATE
SET starts_at = EXCLUDED.starts_at,
    ends_at = EXCLUDED.ends_at,
    status = EXCLUDED.status
WHERE NOT EXISTS (
    SELECT 1
    FROM bookings
    WHERE bookings.slot_id = appointment_slots.id
);

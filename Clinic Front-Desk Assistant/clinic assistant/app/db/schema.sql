-- Bootstrap schema for the clinic front-desk assistant.
-- PostgreSQL 13+ provides gen_random_uuid() without an extension.
-- LangGraph checkpoint tables are created by PostgresSaver.setup() in initialize.py.

CREATE TABLE IF NOT EXISTS doctors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT doctors_full_name_not_blank CHECK (BTRIM(full_name) <> ''),
    CONSTRAINT doctors_specialty_not_blank CHECK (BTRIM(specialty) <> '')
);

CREATE TABLE IF NOT EXISTS appointment_slots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doctor_id UUID NOT NULL REFERENCES doctors (id) ON DELETE RESTRICT,
    starts_at TIMESTAMPTZ NOT NULL,
    ends_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT appointment_slots_valid_time CHECK (ends_at > starts_at),
    CONSTRAINT appointment_slots_valid_status CHECK (status IN ('open', 'blocked')),
    CONSTRAINT appointment_slots_doctor_start_unique UNIQUE (doctor_id, starts_at)
);

-- The thread ID is the stable LangGraph thread identifier for this conversation.
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT conversations_thread_id_not_blank CHECK (BTRIM(thread_id) <> '')
);

CREATE TABLE IF NOT EXISTS bookings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slot_id UUID NOT NULL REFERENCES appointment_slots (id) ON DELETE RESTRICT,
    conversation_id UUID REFERENCES conversations (id) ON DELETE SET NULL,
    -- Store an opaque patient reference, not patient names or clinical details.
    patient_reference TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'confirmed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT bookings_valid_status CHECK (
        status IN ('pending_approval', 'confirmed', 'cancelled', 'rejected')
    ),
    CONSTRAINT bookings_patient_reference_not_blank CHECK (BTRIM(patient_reference) <> '')
);

-- Enforced in PostgreSQL so concurrent requests cannot reserve one slot twice.
CREATE UNIQUE INDEX IF NOT EXISTS bookings_one_active_reservation_per_slot
    ON bookings (slot_id)
    WHERE status IN ('pending_approval', 'confirmed');

CREATE INDEX IF NOT EXISTS appointment_slots_search_idx
    ON appointment_slots (starts_at, doctor_id)
    WHERE status = 'open';

CREATE INDEX IF NOT EXISTS bookings_conversation_idx
    ON bookings (conversation_id)
    WHERE conversation_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id UUID NOT NULL REFERENCES bookings (id) ON DELETE RESTRICT,
    conversation_id UUID NOT NULL REFERENCES conversations (id) ON DELETE RESTRICT,
    action TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ,
    decided_by TEXT,
    decision_note TEXT,
    CONSTRAINT approvals_valid_action CHECK (action IN ('same_day_booking', 'cancel_booking')),
    CONSTRAINT approvals_valid_status CHECK (status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT approvals_decision_timestamp CHECK (
        (status = 'pending' AND decided_at IS NULL)
        OR (status IN ('approved', 'rejected') AND decided_at IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS approvals_one_pending_action_per_booking
    ON approvals (booking_id, action)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS approvals_pending_queue_idx
    ON approvals (requested_at)
    WHERE status = 'pending';

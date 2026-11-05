-- Bootstrap schema for the clinic front-desk assistant.
-- PostgreSQL 13+ provides gen_random_uuid() without an extension.
-- LangGraph checkpoint tables are created by PostgresSaver.setup() in initialize.py.
-- btree_gist enables a GiST exclusion constraint over doctor UUIDs and time ranges.
CREATE EXTENSION IF NOT EXISTS btree_gist;

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
    CONSTRAINT appointment_slots_no_doctor_overlap EXCLUDE USING GIST (
        doctor_id WITH =,
        tstzrange(starts_at, ends_at, '[)') WITH &&
    )
);

-- Migrate an existing bootstrap database as well as initializing a fresh one.
ALTER TABLE appointment_slots
    DROP CONSTRAINT IF EXISTS appointment_slots_doctor_start_unique;

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'appointment_slots_no_doctor_overlap'
          AND conrelid = 'appointment_slots'::regclass
    ) THEN
        ALTER TABLE appointment_slots
            ADD CONSTRAINT appointment_slots_no_doctor_overlap EXCLUDE USING GIST (
                doctor_id WITH =,
                tstzrange(starts_at, ends_at, '[)') WITH &&
            );
    END IF;
END;
$migration$;

-- The thread ID is the stable LangGraph thread identifier for this conversation.
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id TEXT NOT NULL UNIQUE,
    -- Opaque ID from authenticated API context; never a patient name or message.
    owner_reference TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT conversations_thread_id_not_blank CHECK (BTRIM(thread_id) <> ''),
    CONSTRAINT conversations_owner_reference_not_blank
        CHECK (owner_reference IS NULL OR BTRIM(owner_reference) <> '')
);

-- Existing conversations remain unowned until an authenticated migration or service assigns them.
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS owner_reference TEXT;

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversations_owner_reference_not_blank'
          AND conrelid = 'conversations'::regclass
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_owner_reference_not_blank
            CHECK (owner_reference IS NULL OR BTRIM(owner_reference) <> '');
    END IF;
END;
$migration$;

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
    CONSTRAINT bookings_patient_reference_not_blank CHECK (BTRIM(patient_reference) <> ''),
    CONSTRAINT bookings_id_conversation_key UNIQUE (id, conversation_id)
);

-- Keep the composite key available when upgrading a database created by an older schema.
DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'bookings_id_conversation_key'
          AND conrelid = 'bookings'::regclass
    ) THEN
        ALTER TABLE bookings
            ADD CONSTRAINT bookings_id_conversation_key UNIQUE (id, conversation_id);
    END IF;
END;
$migration$;

-- A partial unique index serializes competing reservations for the same slot.
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
    booking_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ,
    decided_by TEXT,
    decision_note TEXT,
    CONSTRAINT approvals_booking_conversation_fk
        FOREIGN KEY (booking_id, conversation_id)
        REFERENCES bookings (id, conversation_id)
        ON DELETE RESTRICT,
    CONSTRAINT approvals_valid_action CHECK (action IN ('same_day_booking', 'cancel_booking')),
    CONSTRAINT approvals_valid_status CHECK (status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT approvals_decision_timestamp CHECK (
        (status = 'pending' AND decided_at IS NULL)
        OR (status IN ('approved', 'rejected') AND decided_at IS NOT NULL)
    )
);

-- Replace the old independent foreign keys with one that enforces conversation ownership.
ALTER TABLE approvals
    DROP CONSTRAINT IF EXISTS approvals_booking_id_fkey;
ALTER TABLE approvals
    DROP CONSTRAINT IF EXISTS approvals_conversation_id_fkey;

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'approvals_booking_conversation_fk'
          AND conrelid = 'approvals'::regclass
    ) THEN
        ALTER TABLE approvals
            ADD CONSTRAINT approvals_booking_conversation_fk
            FOREIGN KEY (booking_id, conversation_id)
            REFERENCES bookings (id, conversation_id)
            ON DELETE RESTRICT;
    END IF;
END;
$migration$;

CREATE UNIQUE INDEX IF NOT EXISTS approvals_one_pending_action_per_booking
    ON approvals (booking_id, action)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS approvals_pending_queue_idx
    ON approvals (requested_at)
    WHERE status = 'pending';

-- Lock the slot and doctor while booking so concurrent block/deactivation updates serialize.
CREATE OR REPLACE FUNCTION enforce_bookable_slot_for_booking()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
DECLARE
    slot_status TEXT;
    doctor_is_active BOOLEAN;
BEGIN
    IF NEW.status NOT IN ('pending_approval', 'confirmed') THEN
        RETURN NEW;
    END IF;

    SELECT slot.status, doctor.is_active
    INTO slot_status, doctor_is_active
    FROM appointment_slots AS slot
    JOIN doctors AS doctor ON doctor.id = slot.doctor_id
    WHERE slot.id = NEW.slot_id
    FOR UPDATE OF slot, doctor;

    IF NOT FOUND
       OR slot_status IS DISTINCT FROM 'open'
       OR doctor_is_active IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'Bookings require an open slot and an active doctor.'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS bookings_require_bookable_slot ON bookings;
CREATE TRIGGER bookings_require_bookable_slot
    BEFORE INSERT OR UPDATE ON bookings
    FOR EACH ROW
    EXECUTE FUNCTION enforce_bookable_slot_for_booking();

-- Do not block, move, or reassign a slot while it has a live reservation.
CREATE OR REPLACE FUNCTION prevent_changing_reserved_slot()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
    IF (
        NEW.status IS DISTINCT FROM 'open'
        OR NEW.doctor_id IS DISTINCT FROM OLD.doctor_id
        OR NEW.starts_at IS DISTINCT FROM OLD.starts_at
        OR NEW.ends_at IS DISTINCT FROM OLD.ends_at
    ) AND EXISTS (
        SELECT 1
        FROM bookings
        WHERE slot_id = OLD.id
          AND status IN ('pending_approval', 'confirmed')
    ) THEN
        RAISE EXCEPTION 'A slot with an active booking cannot be blocked or changed.'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS appointment_slots_protect_active_bookings ON appointment_slots;
CREATE TRIGGER appointment_slots_protect_active_bookings
    BEFORE UPDATE OF status, doctor_id, starts_at, ends_at ON appointment_slots
    FOR EACH ROW
    EXECUTE FUNCTION prevent_changing_reserved_slot();

-- An active doctor cannot be deactivated while any of their slots are reserved.
CREATE OR REPLACE FUNCTION prevent_deactivating_booked_doctor()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
    IF OLD.is_active IS TRUE
       AND NEW.is_active IS FALSE
       AND EXISTS (
           SELECT 1
           FROM appointment_slots AS slot
           JOIN bookings AS booking ON booking.slot_id = slot.id
           WHERE slot.doctor_id = OLD.id
             AND booking.status IN ('pending_approval', 'confirmed')
       ) THEN
        RAISE EXCEPTION 'A doctor with active bookings cannot be deactivated.'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS doctors_protect_active_bookings ON doctors;
CREATE TRIGGER doctors_protect_active_bookings
    BEFORE UPDATE OF is_active ON doctors
    FOR EACH ROW
    EXECUTE FUNCTION prevent_deactivating_booked_doctor();

-- Refresh update timestamps consistently even when callers omit updated_at.
CREATE OR REPLACE FUNCTION refresh_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at := statement_timestamp();
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS conversations_refresh_updated_at ON conversations;
CREATE TRIGGER conversations_refresh_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION refresh_updated_at();

DROP TRIGGER IF EXISTS bookings_refresh_updated_at ON bookings;
CREATE TRIGGER bookings_refresh_updated_at
    BEFORE UPDATE ON bookings
    FOR EACH ROW
    EXECUTE FUNCTION refresh_updated_at();

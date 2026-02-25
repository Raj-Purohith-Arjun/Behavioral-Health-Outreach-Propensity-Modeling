WITH base AS (
    SELECT
        m.member_id,
        m.age,
        m.plan_tier,
        COALESCE(e.prior_engagements, 0) AS prior_engagements,
        COALESCE(o.outreach_count_90d, 0) AS outreach_count_90d,
        COALESCE(c.days_since_last_contact, 999) AS days_since_last_contact,
        COALESCE(s.severity_score, 0) AS severity_score,
        COALESCE(p.podcast_minutes_30d, 0) AS podcast_minutes,
        COALESCE(em.email_open_rate_30d, 0) AS email_open_rate,
        CASE WHEN t.member_id IS NOT NULL THEN 1 ELSE 0 END AS treatment,
        CASE WHEN g.member_id IS NOT NULL THEN 1 ELSE 0 END AS engaged
    FROM members m
    LEFT JOIN engagement_features e ON m.member_id = e.member_id
    LEFT JOIN outreach_features o ON m.member_id = o.member_id
    LEFT JOIN contact_features c ON m.member_id = c.member_id
    LEFT JOIN clinical_features s ON m.member_id = s.member_id
    LEFT JOIN podcast_features p ON m.member_id = p.member_id
    LEFT JOIN email_features em ON m.member_id = em.member_id
    LEFT JOIN treatment_assignments t ON m.member_id = t.member_id
    LEFT JOIN engagement_labels g ON m.member_id = g.member_id
)
SELECT *
FROM base;

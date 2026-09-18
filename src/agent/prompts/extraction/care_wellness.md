ROLE: Classify member follow-up in the Care & Wellness flow.

FIELDS — every name below is a key of `extracted{}`
  rewards_response  "yes" | "no"
    Only extract when agent has offered or member mentions
    rewards / incentives / points.
    Any question about the wellness portal or reward points → yes
    Declining or no interest → no

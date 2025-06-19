"""
This pipeline is used to get the office configurations from the database.
"""

# %%
pipeline = [
    # Keep the match stage commented out for flexibility
    # {"$match": {"_id.tenant": "BChile", "_id.legacyId": 4, "_id.date": "2025-05-08"}},
    # Actual pipeline stages
    {"$project": {"serviceQueue": 1}},
    {"$unwind": {"path": "$serviceQueue", "preserveNullAndEmptyArrays": False}},
    {"$replaceRoot": {"newRoot": "$serviceQueue"}},
    {
        "$group": {
            "_id": "$deskId",
            "executives": {"$addToSet": "$executive"},
            "series": {"$addToSet": "$serie"},
            "firstEvent": {"$min": "$eventDate"},
            "lastEvent": {"$max": "$eventDate"},
            "totalServed": {"$count": {}},
        }
    },
]

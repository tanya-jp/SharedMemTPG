curl -X POST https://www.comet.com/api/rest/v2/write/experiment/create \
     -H "Content-Type: application/json" \
     -H "Authorization: $COMET_API_KEY" \
     -d '{
            "workspaceName": "genetic-programming",
            "projectName": "tpg",
            "experimentName": "test-experiment"
        }'

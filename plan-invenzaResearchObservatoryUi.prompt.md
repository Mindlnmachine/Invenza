## Plan: Invenza Research Observatory UI

Build a connected React application inspired by the provided technical dashboards: dark observatory styling, dense telemetry, interactive patent graph, and a clear grounding-analysis workspace.

**Steps**

1. **Frontend foundation**
   - Create a Vite React/TypeScript app in `Frontend/`.
   - Add package configuration, Dockerfile, API client, typed models, and responsive layout.
   - Make the existing Docker Compose frontend service build successfully.

2. **Research dashboard**
   - Add a dark technical command center with:
     - left navigation
     - topic/query composer
     - API/model status
     - recent investigations
     - system telemetry
   - Connect health status to `GET /health`.

3. **Patent investigation workspace**
   - Connect the query form to `POST /ideas`.
   - Display:
     - generated concept
     - summary
     - source patents
     - grounding score
     - pass/fail target state
     - retrieval metrics
     - selected model
     - inference latency
     - evidence summary

4. **Full patent network**
   - Extend the backend with a graph-data endpoint based on actual PostgreSQL patent metadata and citation relationships.
   - Render an interactive graph with:
     - generated idea node
     - retrieved patent nodes
     - citation/entity relationships where available
     - graph filtering and node selection
     - detail inspector panel
   - Avoid creating relationships that are not present in the database.

5. **Visual system**
   - Use the dark observatory direction from the second reference:
     - near-black background
     - cyan, electric blue, violet, and magenta evidence accents
     - thin technical borders
     - compact telemetry panels
     - graph-focused main workspace
     - responsive mobile layout

6. **Evidence inspector and history**
   - Add patent detail view with available metadata, similarity, assignee, filing date, classification codes, and external link.
   - Add browser-local investigation history initially.
   - Support loading, empty, error, and unavailable-data states.

7. **Validation**
   - Run frontend typecheck and production build.
   - Run backend tests.
   - Verify Docker Compose configuration.
   - Test `/health`, `/ideas`, graph loading, API errors, responsive layout, and score states.

**Relevant files**

- `Frontend/` — new React/TypeScript application
- `Backend/app/api.py` — CORS, graph endpoint, typed API responses
- `Backend/app/pipeline.py` — query limit and structured evidence output
- `Backend/app/retrieval.py` — patent metadata and similarity retrieval
- `Backend/app/grounding.py` — score breakdown displayed in the UI
- `Backend/docker-compose.yml` — frontend service integration
- `patent_images/` — optional patent visual evidence assets

**Initial screens**

- Research Command Center
- Investigation / Patent Network
- Grounding Analysis
- Patent Evidence Inspector
- Local Investigation History
- System Status

**Important implementation decision**

The full network view cannot be implemented honestly from the current `/ideas` response alone because it only returns patent IDs. The backend must expose real graph data from stored citation and metadata fields before displaying patent-to-patent relationships.

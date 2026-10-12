import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import './App.css'

type ChatRole = 'user' | 'assistant'
type ChatMessage = { role: ChatRole; content: string }
type HistoryEntry = {
  title: string
  summary: string
  message: string
  at: string
  conversation: ChatMessage[]
  idea: IdeaResult
}

type Patent = {
  id: string
  title: string
  cluster: string
  domain: string
  color: string
  x: number
  y: number
}

type IdeaResult = {
  concept_name: string
  summary: string
  source_patents: string[]
  used_patent_ids: string[]
  grounding_score: number
  model_used: string
  inference_time_ms: number
}

const domainPalette: Record<string, string> = {
  agriculture: '#6ee7b7',
  'artificial-intelligence': '#7dd3fc',
  chemical: '#f9a8d4',
  sensors: '#c4b5fd',
}

const defaultPatents: Patent[] = [
  { id: 'US-10298744', title: 'Adaptive irrigation scheduling', cluster: 'Agriculture', domain: 'agriculture', color: domainPalette.agriculture, x: 170, y: 110 },
  { id: 'EP-4412201', title: 'Drip-control telemetry mesh', cluster: 'Sensors', domain: 'sensors', color: domainPalette.sensors, x: 395, y: 170 },
  { id: 'WO-20199512', title: 'Soil moisture forecasting', cluster: 'AI', domain: 'artificial-intelligence', color: domainPalette['artificial-intelligence'], x: 265, y: 290 },
  { id: 'US-11003201', title: 'Low-power sensor nodes', cluster: 'Chemical', domain: 'chemical', color: domainPalette.chemical, x: 520, y: 130 },
  { id: 'US-8764571', title: 'Crop health analytics', cluster: 'Agriculture', domain: 'agriculture', color: domainPalette.agriculture, x: 610, y: 300 },
  { id: 'CN-11836119', title: 'Pest-driven irrigation loop', cluster: 'Sensors', domain: 'sensors', color: domainPalette.sensors, x: 470, y: 430 },
  { id: 'US-11900162', title: 'Autonomous pump optimization', cluster: 'AI', domain: 'artificial-intelligence', color: domainPalette['artificial-intelligence'], x: 720, y: 260 },
]

const defaultIdea: IdeaResult = {
  concept_name: 'Awaiting research direction',
  summary: 'Submit a research question to generate a grounded business concept from the patent landscape.',
  source_patents: [],
  used_patent_ids: [],
  grounding_score: 0,
  model_used: '—',
  inference_time_ms: 0,
}

const emptyPatent: Patent = {
  id: 'unavailable',
  title: 'Patent cluster unavailable',
  cluster: 'Research',
  domain: 'research',
  color: '#7dd3fc',
  x: 410,
  y: 210,
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

type PatentCluster3DProps = {
  patents: Patent[]
  usedPatentIds: Set<string>
  selectedPatentId: string
  onSelect: (patentId: string) => void
  zoom: number
}

function PatentCluster3D({ patents, usedPatentIds, selectedPatentId, onSelect, zoom }: PatentCluster3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container || patents.length === 0) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 1, 2000)
    camera.position.set(0, 0, 720 / zoom)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x071017, 0)
    container.appendChild(renderer.domElement)

    const cluster = new THREE.Group()
    scene.add(cluster)
    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const nodeMeshes: THREE.Mesh[] = []
    const dragging = { active: false, x: 0, y: 0 }
    let frame = 0

    const hash = (value: string) => {
      let result = 2166136261
      for (let index = 0; index < value.length; index += 1) {
        result ^= value.charCodeAt(index)
        result = Math.imul(result, 16777619)
      }
      return (result >>> 0) / 4294967295
    }

    const makeLabel = (text: string, color: string, opacity: number) => {
      const canvas = document.createElement('canvas')
      canvas.width = 256
      canvas.height = 44
      const context = canvas.getContext('2d')
      if (!context) return null
      context.font = 'bold 16px sans-serif'
      context.fillStyle = color
      context.globalAlpha = opacity
      context.textAlign = 'center'
      context.textBaseline = 'middle'
      context.fillText(text, canvas.width / 2, canvas.height / 2)
      const texture = new THREE.CanvasTexture(canvas)
      texture.colorSpace = THREE.SRGBColorSpace
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }))
      sprite.scale.set(58, 10, 1)
      return sprite
    }

    patents.forEach((patent) => {
      const x = ((patent.x - 410) / 410) * 270 + (hash(`${patent.id}-x`) - 0.5) * 24
      const y = -((patent.y - 210) / 210) * 150 + (hash(`${patent.id}-y`) - 0.5) * 20
      const z = (hash(`${patent.id}-z`) - 0.5) * 210
      const isUsed = usedPatentIds.has(patent.id)
      const isSelected = selectedPatentId === patent.id
      const material = new THREE.MeshBasicMaterial({
        color: isUsed || isSelected ? patent.color : '#27445d',
        transparent: true,
        opacity: isUsed || isSelected ? 1 : 0.7,
      })
      const node = new THREE.Mesh(new THREE.SphereGeometry(isUsed || isSelected ? 5.5 : 3, 8, 8), material)
      node.position.set(x, y, z)
      node.userData.patentId = patent.id
      cluster.add(node)
      nodeMeshes.push(node)

      const label = makeLabel(patent.id, isUsed || isSelected ? '#f5fbff' : patent.color, isUsed || isSelected ? 1 : 0.28)
      if (label) {
        label.position.set(x, y + 9, z)
        cluster.add(label)
      }
    })

    const resize = () => {
      const width = container.clientWidth
      const height = container.clientHeight
      camera.aspect = width / Math.max(height, 1)
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }
    const onPointerDown = (event: PointerEvent) => {
      dragging.active = true
      dragging.x = event.clientX
      dragging.y = event.clientY
      renderer.domElement.setPointerCapture(event.pointerId)
    }
    const onPointerMove = (event: PointerEvent) => {
      if (dragging.active) {
        cluster.rotation.y += (event.clientX - dragging.x) * 0.006
        cluster.rotation.x += (event.clientY - dragging.y) * 0.004
        dragging.x = event.clientX
        dragging.y = event.clientY
      }
    }
    const onPointerUp = (event: PointerEvent) => {
      dragging.active = false
      renderer.domElement.releasePointerCapture(event.pointerId)
    }
    const onClick = (event: MouseEvent) => {
      const bounds = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const hit = raycaster.intersectObjects(nodeMeshes)[0]
      if (hit?.object.userData.patentId) onSelect(hit.object.userData.patentId)
    }

    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('click', onClick)
    window.addEventListener('resize', resize)
    resize()

    const animate = () => {
      cluster.rotation.y += dragging.active ? 0.0008 : 0.0025
      cluster.rotation.x = Math.sin(performance.now() * 0.00025) * 0.08
      renderer.render(scene, camera)
      frame = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('click', onClick)
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Sprite) {
          object.geometry?.dispose()
          const material = object.material as THREE.Material & { map?: THREE.Texture }
          material.map?.dispose()
          material.dispose()
        }
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [onSelect, patents, selectedPatentId, usedPatentIds, zoom])

  return <div ref={containerRef} className="patent-cluster-3d" role="img" aria-label="Rotating three-dimensional patent cluster" />
}

function App() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [idea, setIdea] = useState<IdeaResult>(defaultIdea)
  const [patents, setPatents] = useState<Patent[]>(defaultPatents)
  const [selectedPatentId, setSelectedPatentId] = useState(defaultPatents[0].id)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [clusterError, setClusterError] = useState('')
  const [clusterZoom, setClusterZoom] = useState(1)

  const loadPatentCluster = async (topic: string) => {
    try {
      const response = await fetch(`${API_BASE}/patent-cluster?limit=500`)

      if (!response.ok) {
        throw new Error('Failed to load patent cluster')
      }

      const data = await response.json()
      const nextPatents = Array.isArray(data.nodes) && data.nodes.length > 0
        ? data.nodes.map((node: any) => ({
            id: String(node.id),
            title: String(node.title || node.id),
            cluster: String(node.cluster || node.domain || 'Research'),
            domain: String(node.domain || 'agriculture'),
            color: String(node.color || domainPalette[node.domain] || '#7dd3fc'),
            x: Number(node.x ?? 220),
            y: Number(node.y ?? 200),
          }))
        : []

      if (nextPatents.length === 0) {
        throw new Error('The patent cluster returned no nodes')
      }

      setPatents(nextPatents)
      setSelectedPatentId(nextPatents[0].id)
      setClusterError('')

      if (!topic) return

      const ideaResponse = await fetch(`${API_BASE}/ideas`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, top_k: 8 }),
      })

      if (ideaResponse.ok) {
        const ideaData = await ideaResponse.json()
        setIdea({
          concept_name: ideaData.concept_name || defaultIdea.concept_name,
          summary: ideaData.summary || defaultIdea.summary,
          source_patents: Array.isArray(ideaData.source_patents) ? ideaData.source_patents : defaultIdea.source_patents,
          used_patent_ids: Array.isArray(ideaData.used_patent_ids) ? ideaData.used_patent_ids : defaultIdea.used_patent_ids,
          grounding_score: Number(ideaData.grounding_score ?? defaultIdea.grounding_score),
          model_used: ideaData.model_used || defaultIdea.model_used,
          inference_time_ms: Number(ideaData.inference_time_ms ?? defaultIdea.inference_time_ms),
        })
      }
    } catch {
      setPatents(defaultPatents)
      setSelectedPatentId(defaultPatents[0].id)
      setClusterError(`Live patent data is unavailable. Showing the sample cluster until ${API_BASE} is online.`)
    }
  }

  useEffect(() => {
    const stored = localStorage.getItem('invenza-history')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        const entries = Array.isArray(parsed) ? parsed.map((item): HistoryEntry | null => {
          try {
            const entry = typeof item === 'string' ? JSON.parse(item) : item
            if (!entry?.title || !entry?.message) return null
            const conversation = Array.isArray(entry.conversation)
              ? entry.conversation
              : [{ role: 'user' as const, content: entry.message }]
            return {
              title: String(entry.title),
              summary: String(entry.summary || ''),
              message: String(entry.message),
              at: String(entry.at || new Date().toISOString()),
              conversation,
              idea: entry.idea || defaultIdea,
            }
          } catch {
            return null
          }
        }).filter((entry): entry is HistoryEntry => entry !== null) : []
        setHistory(entries)
      } catch {
        setHistory([])
      }
    }
  }, [])

  useEffect(() => {
    void loadPatentCluster('')
  }, [])

  const usedPatentSet = useMemo(() => new Set(idea.used_patent_ids), [idea.used_patent_ids])

  const saveHistory = (nextIdea: IdeaResult, nextMessage: string, conversation: ChatMessage[]) => {
    const item: HistoryEntry = {
      title: nextIdea.concept_name,
      summary: nextIdea.summary,
      message: nextMessage,
      at: new Date().toISOString(),
      conversation,
      idea: nextIdea,
    }

    const nextHistory = [item, ...history].slice(0, 8)
    setHistory(nextHistory)
    localStorage.setItem('invenza-history', JSON.stringify(nextHistory))
  }

  const deleteHistoryItem = (indexToDelete: number) => {
    const nextHistory = history.filter((_, index) => index !== indexToDelete)
    setHistory(nextHistory)
    localStorage.setItem('invenza-history', JSON.stringify(nextHistory))
  }

  const openHistoryItem = (entry: HistoryEntry) => {
    setMessages(entry.conversation)
    setIdea(entry.idea)
    setInput('')
    setClusterError('')
    const preferredPatent = entry.idea.used_patent_ids[0] || entry.idea.source_patents[0]
    if (preferredPatent && patents.some((patent) => patent.id === preferredPatent)) {
      setSelectedPatentId(preferredPatent)
    }
  }

  const startNewQuery = () => {
    setMessages([])
    setInput('')
    setIdea(defaultIdea)
    setSelectedPatentId(patents[0]?.id || defaultPatents[0].id)
    setClusterError('')
  }

  const selectedPatent = patents.find((patent) => patent.id === selectedPatentId) ?? patents[0] ?? emptyPatent

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const query = input.trim()
    if (!query) return

    const nextUser = { role: 'user' as const, content: query }
    const nextConversation = [...messages, nextUser]
    setInput('')
    setMessages(nextConversation)
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: query, conversation: nextConversation, top_k: 4 }),
      })

      if (!response.ok) {
        const errorBody = await response.text()
        throw new Error(errorBody || `Backend returned ${response.status}`)
      }

      const data = await response.json()
      const assistantContent = data.response || data.summary
      if (!assistantContent) {
        throw new Error('The backend returned an empty answer.')
      }
      const updatedConversation = [...nextConversation, { role: 'assistant' as const, content: assistantContent }]
      setMessages(updatedConversation)

      const responsePatentIds = [
        ...(Array.isArray(data.used_patent_ids) ? data.used_patent_ids : []),
        ...(Array.isArray(data.source_patents) ? data.source_patents : []),
        ...(Array.isArray(data.evidence) ? data.evidence.map((item: { patent_id?: string }) => item.patent_id) : []),
      ].filter((patentId): patentId is string => Boolean(patentId)).map((patentId) => String(patentId).trim())
      const highlightedPatentIds = [...new Set(responsePatentIds)]

      const nextIdea: IdeaResult = {
        concept_name: data.concept_name || 'New concept',
        summary: data.summary || assistantContent,
        source_patents: Array.isArray(data.source_patents) ? data.source_patents.map(String) : defaultIdea.source_patents,
        used_patent_ids: highlightedPatentIds,
        grounding_score: Number(data.grounding_score ?? 0),
        model_used: data.model_used || 'llama3.2',
        inference_time_ms: Number(data.inference_time_ms ?? 0),
      }

      if (patents.length > 0) {
        const preferred = highlightedPatentIds[0] || patents[0].id
        setSelectedPatentId(preferred)
      }

      const evidencePatents = Array.isArray(data.evidence) ? data.evidence : []
      if (evidencePatents.length > 0) {
        setPatents((currentPatents) => {
          const existingIds = new Set(currentPatents.map((patent) => patent.id))
          const missingPatents = evidencePatents
            .map((evidence: { patent_id?: string; title?: string }, index: number) => {
              const id = String(evidence.patent_id || '').trim()
              if (!id || existingIds.has(id)) return null
              return {
                id,
                title: evidence.title || id,
                cluster: 'Response reference',
                domain: 'research',
                color: '#facc15',
                x: 330 + index * 70,
                y: 210 + (index % 2) * 70,
              }
            })
            .filter((patent: Patent | null): patent is Patent => patent !== null)
          return missingPatents.length > 0 ? [...currentPatents, ...missingPatents] : currentPatents
        })
      }

      setIdea(nextIdea)
      saveHistory(nextIdea, query, updatedConversation)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown backend error.'
      const fallbackConversation = [...nextConversation, { role: 'assistant' as const, content: `I could not answer that query. ${errorMessage}` }]
      setMessages(fallbackConversation)
      setClusterError(`The backend did not respond. Start it at ${API_BASE} to keep the full cluster visible.`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleExport = async () => {
    try {
      const response = await fetch(`${API_BASE}/export/docx`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(idea),
      })

      if (!response.ok) {
        throw new Error('Export failed')
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${idea.concept_name.toLowerCase().replace(/\s+/g, '-')}.docx`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch {
      setClusterError(`Export failed. Start the backend at ${API_BASE} and try again.`)
    }
  }

  return (
    <div className="observatory-shell">
      <header className="topbar">
        <div className="status-group">
          <span className="time">9:04</span>
          <span className="signal">◔</span>
        </div>
        <div className="status-icons">
          <span>Vo</span>
          <span>LTE</span>
          <span>25</span>
        </div>
      </header>

      <main className="main-panel">
        <aside className="sidebar">
          <div className="sidebar-header">History</div>
          <ul className="history-list">
            {history.length > 0 ? (
              history.map((item, index) => (
                  <li key={`${item.at}-${index}`}>
                    <div className="history-item-header">
                      <button className="history-open" type="button" onClick={() => openHistoryItem(item)}>
                        <strong>{item.title}</strong>
                      </button>
                      <button
                        className="history-delete"
                        type="button"
                        aria-label={`Delete ${item.title}`}
                        title="Delete history item"
                        onClick={(event) => {
                          event.stopPropagation()
                          deleteHistoryItem(index)
                        }}
                      >
                        ×
                      </button>
                    </div>
                    <button className="history-open history-message" type="button" onClick={() => openHistoryItem(item)}>
                      {item.message}
                    </button>
                  </li>
              ))
            ) : (
              <li>Saved ideas will appear here.</li>
            )}
          </ul>
        </aside>

        <section className="workspace-panel">
          <div className="panel-toolbar">
            <div className="toolbar-group">
              <button className="chip active" type="button">Patent Cluster</button>
              <button className="chip" type="button">Narrative</button>
              <button className="chip" type="button">Grounding</button>
            </div>
            <div className="toolbar-actions">
              <div className="zoom-controls" aria-label="Cluster zoom controls">
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Zoom out"
                  title="Zoom out"
                  onClick={() => setClusterZoom((value) => Math.max(0.65, Number((value - 0.15).toFixed(2))))}
                >
                  −
                </button>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Zoom in"
                  title="Zoom in"
                  onClick={() => setClusterZoom((value) => Math.min(2.2, Number((value + 0.15).toFixed(2))))}
                >
                  +
                </button>
              </div>
              <button className="export-button" type="button" onClick={handleExport}>Export DOCX</button>
            </div>
          </div>

          <div className="cluster-stage">
            {clusterError && <div className="cluster-error">{clusterError}</div>}
            <PatentCluster3D
              patents={patents}
              usedPatentIds={usedPatentSet}
              selectedPatentId={selectedPatentId}
              onSelect={setSelectedPatentId}
              zoom={clusterZoom}
            />
          </div>

          <div className="metrics-row">
            <div className="metric-card accent">
              <label>Grounding</label>
              <strong>{idea.grounding_score.toFixed(1)}</strong>
            </div>
            <div className="metric-card">
              <label>Model</label>
              <strong>{idea.model_used}</strong>
            </div>
            <div className="metric-card">
              <label>Latency</label>
              <strong>{idea.inference_time_ms} ms</strong>
            </div>
            <div className="metric-card">
              <label>Patents</label>
              <strong>{patents.length}</strong>
            </div>
          </div>
        </section>

        <aside className="chat-panel">
          <div className="chat-header">
            <span>Research assistant</span>
            <div className="chat-header-actions">
              <button className="mini-button" type="button" onClick={startNewQuery}>New query</button>
              <button className="mini-button" type="button">Live</button>
            </div>
          </div>

          <div className="chat-scroll">
            {messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
                {message.content}
              </div>
            ))}
            {isLoading && <div className="chat-bubble assistant loading">Thinking…</div>}
          </div>

          <form onSubmit={handleSubmit} className="composer">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              rows={4}
              placeholder="Ask your Question"
            />
            <div className="composer-actions">
              <button type="button" className="secondary-button" onClick={handleExport}>Export</button>
              <button type="submit" className="primary-button" disabled={isLoading}>
                {isLoading ? 'Running…' : 'Submit'}
              </button>
            </div>
          </form>
        </aside>
      </main>

      <section className="inspector-row">
        <div className="inspector-card">
          <div className="inspector-title">Active concept</div>
          <h2>{idea.concept_name}</h2>
          <p>{idea.summary}</p>
        </div>

        <div className="inspector-card compact">
          <div className="inspector-title">Patent details</div>
          <h3>{selectedPatent.title}</h3>
          <p>{selectedPatent.id}</p>
          <div className="badge-row">
            <span className="badge">{selectedPatent.cluster}</span>
            <span className="badge muted">{usedPatentSet.has(selectedPatent.id) ? 'Used in answer' : 'Context node'}</span>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App

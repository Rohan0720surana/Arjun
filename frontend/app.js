/* ===========================================================================
   ARJUN — dashboard
   ---------------------------------------------------------------------------
   No build step and no framework: one script, served as-is by FastAPI, so the
   thing a judge or a security office runs is the thing in the repository.

   Three ideas worth knowing:

   · Each camera tile owns its own WebSocket and closes it the moment the tile
     stops being visible, so a wall of twelve cameras in a browser tab nobody
     is looking at costs nothing.
   · Frames arrive as binary JPEG and are painted through createImageBitmap
     onto a canvas. Object URLs per frame would leak unless revoked by hand.
   · The chime is synthesised with the Web Audio API rather than shipped as an
     asset — and the Signal for Help alert is deliberately exempt from it.
   ========================================================================= */

(() => {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    config: null,
    cameras: new Map(),      // name -> { status, tile, socket, raw }
    events: [],
    zones: [],
    token: sessionStorage.getItem("arjun.token") || "",
    muted: localStorage.getItem("arjun.muted") === "1",
    filters: { camera: "", type: "", severity: "" },
    charts: { types: null, crowd: null },
    lastAlertId: 0,
  };

  const api = {
    async get(path) {
      const res = await fetch(path, { headers: authHeaders() });
      if (!res.ok) throw new Error((await safeJson(res))?.error || res.statusText);
      return res.json();
    },
    async send(path, body, method = "POST") {
      const res = await fetch(path, {
        method,
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      if (!res.ok) throw new Error((await safeJson(res))?.error || res.statusText);
      return res.status === 204 ? null : res.json();
    },
  };

  function authHeaders() {
    return state.token ? { "X-Admin-Token": state.token } : {};
  }
  async function safeJson(res) { try { return await res.json(); } catch { return null; } }

  const fmt = {
    clock: (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false }),
    when: (ts) => new Date(ts * 1000).toLocaleString([], {
      day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
      second: "2-digit", hour12: false,
    }),
    bytes: (n) => n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB`
                : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB`
                : `${Math.max(0, Math.round(n / 1e3))} kB`,
  };

  /* =======================================================================
     Boot
     ===================================================================== */

  async function boot() {
    wireChrome();
    try {
      state.config = await api.get("/api/config");
      applyConfig();
    } catch (err) {
      toast({ title: "Cannot reach the server", message: String(err),
              severity: "critical" });
      return;
    }
    await Promise.allSettled([
      refreshStatus(), refreshEvents(), refreshPrivacy(),
      refreshAnalytics(), refreshRules(),
    ]);
    connectEvents();
    setInterval(refreshPrivacy, 20000);
    setInterval(refreshAnalytics, 30000);
    setInterval(refreshRules, 15000);
  }

  function applyConfig() {
    const c = state.config;
    $("#colophon-version").textContent = `v${c.app.version}`;
    $("#colophon-venue").textContent = c.app.venue;
    document.title = `${c.app.name} · ${c.app.subtitle}`;
    setDemo(c.demo_mode, false);

    const assistant = c.assistant || {};
    $("#ask-engine").textContent = assistant.configured
      ? `answered by ${assistant.model}`
      : "offline search — add CLAUDE_API_KEY for written answers";

    const typeFilter = $("#filter-type");
    Object.entries(c.alert_specs).forEach(([key, spec]) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = spec.title;
      typeFilter.append(opt);
    });

    $("#mute-label").textContent = state.muted ? "Sound off" : "Sound on";
    $("#mute-toggle").setAttribute("aria-pressed", String(state.muted));
  }

  function wireChrome() {
    $("#demo-toggle").addEventListener("click", async () => {
      const next = $("#demo-toggle").getAttribute("aria-checked") !== "true";
      try {
        const res = await api.send("/api/demo", { on: next });
        setDemo(res.demo_mode, true);
      } catch (err) {
        toast({ title: "Could not switch mode", message: String(err),
                severity: "warning" });
      }
    });

    $("#mute-toggle").addEventListener("click", () => {
      state.muted = !state.muted;
      localStorage.setItem("arjun.muted", state.muted ? "1" : "0");
      $("#mute-label").textContent = state.muted ? "Sound off" : "Sound on";
      $("#mute-toggle").setAttribute("aria-pressed", String(state.muted));
    });

    ["camera", "type", "severity"].forEach((key) => {
      $(`#filter-${key}`).addEventListener("change", (e) => {
        state.filters[key] = e.target.value;
        renderTimeline();
      });
    });

    $("#ask-form").addEventListener("submit", (e) => {
      e.preventDefault();
      ask($("#ask-input").value);
    });
    $$("#ask-suggestions button").forEach((btn) => {
      btn.addEventListener("click", () => {
        $("#ask-input").value = btn.textContent;
        ask(btn.textContent);
      });
    });

    $("#analytics-range").addEventListener("change", refreshAnalytics);
    $("#zone-close").addEventListener("click", () => $("#zone-dialog").close());
    $("#event-close").addEventListener("click", () => $("#event-dialog").close());
  }

  function setDemo(on, announce) {
    const btn = $("#demo-toggle");
    btn.setAttribute("aria-checked", String(!!on));
    if (announce) {
      toast({
        title: on ? "Demo timings on" : "Live timings restored",
        message: on
          ? "Every waiting period is shortened. The rules themselves are unchanged."
          : "Loitering, unattended bags and the rest are back to realistic times.",
        severity: "info",
      });
    }
  }

  /* =======================================================================
     Live link
     ===================================================================== */

  function connectEvents() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
    const link = $("#link-state");

    let keepalive = 0;
    ws.addEventListener("open", () => {
      link.dataset.state = "live";
      link.textContent = "Live";
      // The server only notices a dead client when a read fails, so say hello.
      keepalive = setInterval(() => { if (ws.readyState === 1) ws.send("."); }, 25000);
    });
    ws.addEventListener("close", () => {
      clearInterval(keepalive);      // or every reconnect leaves one behind
      link.dataset.state = "lost";
      link.textContent = "Reconnecting";
      setTimeout(connectEvents, 2500);
    });
    ws.addEventListener("error", () => ws.close());
    ws.addEventListener("message", (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      handle(msg);
    });
  }

  function handle(msg) {
    switch (msg.kind) {
      case "hello":
      case "status":
        applyStatus(msg.data);
        break;
      case "alert":
        onAlert(msg.data);
        break;
      case "alert.update":
        onAlertUpdate(msg.data);
        break;
      case "audit":
        refreshPrivacy();
        break;
      case "demo":
        setDemo(msg.data.on, false);
        break;
      case "camera":
        applyCameraStatus(msg.data);
        break;
    }
  }

  async function refreshStatus() {
    applyStatus(await api.get("/api/status"));
  }

  function applyStatus(data) {
    $("#stat-clock").textContent = data.clock;
    $("#stat-date").textContent = data.date;
    $("#stat-cameras").textContent = `${data.totals.online}/${data.totals.cameras}`;
    $("#stat-people").textContent = data.totals.people;
    $("#stat-fps").textContent = Math.round(data.totals.fps);

    const today = data.today || {};
    $("#stat-alerts").textContent = today.total || 0;
    $("#stat-alerts").closest(".reading")
      .classList.toggle("reading--alarm", (today.critical || 0) > 0);

    (data.cameras || []).forEach(applyCameraStatus);
    syncCameraFilter(data.cameras || []);
  }

  function syncCameraFilter(cameras) {
    const sel = $("#filter-camera");
    if (sel.options.length - 1 === cameras.length) return;
    const current = sel.value;
    sel.innerHTML = '<option value="">Every camera</option>';
    cameras.forEach((cam) => {
      const opt = document.createElement("option");
      opt.value = cam.name;
      opt.textContent = cam.name;
      sel.append(opt);
    });
    sel.value = current;
  }

  /* =======================================================================
     Camera tiles
     ===================================================================== */

  function applyCameraStatus(status) {
    let entry = state.cameras.get(status.name);
    if (!entry) {
      entry = { status, tile: buildTile(status), socket: null, raw: false };
      state.cameras.set(status.name, entry);
      $("#camera-empty")?.remove();
      const grid = $("#camera-grid");
      entry.tile.style.setProperty("--delay", `${state.cameras.size * 70}ms`);
      grid.append(entry.tile);
      observeTile(entry);
    }
    entry.status = status;
    paintTile(entry);
  }

  function buildTile(status) {
    const tile = document.createElement("article");
    tile.className = "tile";
    tile.dataset.camera = status.name;
    tile.innerHTML = `
      <header class="tile__head">
        <span class="pip" data-state="${status.state}"></span>
        <h3 class="tile__name"></h3>
        <span class="badge badge--sim" hidden>Simulated</span>
        <span class="badge badge--raw" hidden>Raw</span>
        <span class="tile__meta"></span>
      </header>
      <div class="tile__stage">
        <canvas></canvas>
        <div class="tile__status">Connecting</div>
      </div>
      <footer class="tile__foot">
        <button class="chip chip--danger" data-act="raw" aria-pressed="false">Raw view</button>
        <button class="chip" data-act="exam_mode" aria-pressed="false">Exam mode</button>
        <button class="chip" data-act="heatmap" aria-pressed="false">Heatmap</button>
        <button class="chip" data-act="after_hours_override" aria-pressed="false">Night mode</button>
        <button class="chip" data-act="zones">Zones</button>
      </footer>`;
    $(".tile__name", tile).textContent = status.name;

    $$("[data-act]", tile).forEach((btn) => {
      btn.addEventListener("click", () => onTileAction(status.name, btn));
    });
    return tile;
  }

  function paintTile(entry) {
    const { status, tile } = entry;
    $(".pip", tile).dataset.state = status.state;
    $(".badge--sim", tile).hidden = !status.simulated;
    $(".badge--raw", tile).hidden = !entry.raw;
    tile.classList.toggle("tile--raw", entry.raw);

    const fps = status.fps ? `${status.fps.toFixed(1)} fps` : "";
    const people = `${status.people} present`;
    $(".tile__meta", tile).textContent = [people, fps].filter(Boolean).join("  ·  ");

    const overlay = $(".tile__status", tile);
    if (status.state === "live") {
      overlay.hidden = true;
    } else {
      overlay.hidden = false;
      overlay.textContent = status.state === "reconnecting"
        ? "Reconnecting…"
        : status.state === "error" ? (status.error || "Camera error")
        : status.state === "paused" ? "Paused" : "Connecting…";
    }

    Object.entries(status.flags || {}).forEach(([key, value]) => {
      const btn = $(`[data-act="${key}"]`, tile);
      if (btn) btn.setAttribute("aria-pressed", String(!!value));
    });
  }

  /* ---- frames ---------------------------------------------------------- */

  function observeTile(entry) {
    // Only stream a tile that is actually on screen.
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) openStream(entry);
        else closeStream(entry);
      });
    }, { rootMargin: "180px" });
    io.observe(entry.tile);
    entry.observer = io;
  }

  function openStream(entry, { raw = entry.raw } = {}) {
    closeStream(entry);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const name = encodeURIComponent(entry.status.name);
    const url = raw
      ? `${proto}://${location.host}/ws/raw/${name}?token=${encodeURIComponent(state.token)}`
      : `${proto}://${location.host}/ws/video/${name}`;

    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    entry.socket = ws;
    entry.raw = raw;

    const canvas = $("canvas", entry.tile);
    const ctx = canvas.getContext("2d", { alpha: false });
    let painting = false;

    ws.addEventListener("message", async (ev) => {
      if (painting) return;                 // drop, never queue
      painting = true;
      try {
        const bitmap = await createImageBitmap(new Blob([ev.data], { type: "image/jpeg" }));
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
        }
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        $(".tile__status", entry.tile).hidden = true;
      } catch { /* a malformed frame is not worth a message */ }
      painting = false;
    });

    ws.addEventListener("close", () => {
      if (entry.socket === ws) entry.socket = null;
      if (raw && entry.raw) {               // the token expired or was revoked
        entry.raw = false;
        paintTile(entry);
        openStream(entry, { raw: false });
      }
    });
  }

  function closeStream(entry) {
    if (entry.socket) {
      try { entry.socket.close(); } catch { /* already gone */ }
      entry.socket = null;
    }
  }

  /* ---- tile actions ---------------------------------------------------- */

  async function onTileAction(camera, btn) {
    const entry = state.cameras.get(camera);
    const act = btn.dataset.act;

    if (act === "zones") return openZoneEditor(camera);

    if (act === "raw") {
      if (entry.raw) {
        await api.send(`/api/lock/${encodeURIComponent(camera)}`, {});
        entry.raw = false;
        paintTile(entry);
        openStream(entry, { raw: false });
        return;
      }
      const ok = await unlock(camera);
      if (!ok) return;
      entry.raw = true;
      paintTile(entry);
      openStream(entry, { raw: true });
      toast({
        title: "Raw view open",
        message: `This is now recorded in the privacy ledger against ${camera}.`,
        severity: "warning",
      });
      refreshPrivacy();
      return;
    }

    const value = btn.getAttribute("aria-pressed") !== "true";
    try {
      const res = await api.send(
        `/api/camera/${encodeURIComponent(camera)}/flag`, { key: act, value });
      Object.entries(res.flags).forEach(([key, v]) => {
        const b = $(`[data-act="${key}"]`, entry.tile);
        if (b) b.setAttribute("aria-pressed", String(!!v));
      });
    } catch (err) {
      toast({ title: "Could not change that", message: String(err),
              severity: "warning" });
    }
  }

  /* =======================================================================
     PIN
     ===================================================================== */

  function unlock(camera) {
    return new Promise((resolve) => {
      const dialog = $("#pin-dialog");
      const input = $("#pin-input");
      const error = $("#pin-error");
      input.value = "";
      error.hidden = true;
      $("#pin-reason").textContent = camera
        ? `Raw footage from ${camera} shows faces. Opening it is recorded in `
          + `the privacy ledger, with the time and the camera.`
        : `This is the only part of the system that shows faces. Opening it is `
          + `recorded in the privacy ledger.`;

      const form = $("#pin-form");
      const onSubmit = async (e) => {
        if (e.submitter?.value !== "ok") { cleanup(); resolve(false); return; }
        e.preventDefault();
        try {
          const res = await api.send("/api/unlock",
            { pin: input.value, camera: camera || "" });
          state.token = res.token;
          sessionStorage.setItem("arjun.token", res.token);
          cleanup();
          dialog.close();
          resolve(true);
        } catch (err) {
          error.textContent = String(err).replace("Error: ", "");
          error.hidden = false;
          input.select();
        }
      };
      const cleanup = () => form.removeEventListener("submit", onSubmit);
      form.addEventListener("submit", onSubmit);
      dialog.addEventListener("close", () => { cleanup(); resolve(false); },
                              { once: true });
      dialog.showModal();
      input.focus();
    });
  }

  /* =======================================================================
     Observations
     ===================================================================== */

  async function refreshEvents() {
    const data = await api.get("/api/events?limit=200");
    state.events = data.events;
    state.lastAlertId = Math.max(0, ...state.events.map((e) => e.id));
    renderTimeline();
  }

  function onAlert(alert) {
    state.events.unshift({
      ...alert,
      clock: alert.clock,
      zone_name: alert.zone_name,
      acknowledged: 0,
      has_clip: alert.clip_state === "pending" || alert.clip_state === "ready",
    });
    state.events = state.events.slice(0, 300);
    renderTimeline();

    const entry = state.cameras.get(alert.camera);
    if (entry && alert.severity !== "info") {
      entry.tile.dataset.alarm = alert.severity;
      setTimeout(() => { delete entry.tile.dataset.alarm; }, 4200);
    }

    // Only interrupt for something that wants attention. Occupancy changes
    // and exam notes still land in Observations; a counting line on a busy
    // doorway would otherwise bury the camera wall under its own receipts.
    if (alert.severity !== "info") {
      toast({
        title: alert.title,
        message: alert.message,
        severity: alert.severity,
        onClick: () => openEvent(alert.id),
      });
    }

    // The SOS alert is silent on purpose: a chime would announce to the room
    // that somebody just asked for help.
    if (!state.muted && alert.severity !== "info" && !alert.silent) {
      chime(alert.severity === "critical");
    }
  }

  function onAlertUpdate(patch) {
    const found = state.events.find((e) => e.id === patch.id);
    if (!found) return;
    Object.assign(found, patch);
    if (patch.clip_state === "ready") found.has_clip = true;
    renderTimeline();
  }

  function renderTimeline() {
    const list = $("#timeline");
    const { camera, type, severity } = state.filters;
    const rank = { info: 0, warning: 1, critical: 2 };

    const rows = state.events.filter((e) => {
      if (camera && e.camera !== camera) return false;
      if (type && e.type !== type) return false;
      if (severity === "warning" && rank[e.severity] < 1) return false;
      if (severity && severity !== "warning" && e.severity !== severity) return false;
      return true;
    });

    $("#obs-count").textContent = rows.length;
    list.innerHTML = "";

    if (!rows.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = state.events.length
        ? "Nothing matches those filters."
        : "Nothing has happened yet. That is the normal state.";
      list.append(li);
      return;
    }

    rows.slice(0, 120).forEach((e) => {
      const li = document.createElement("li");
      li.className = "entry";
      li.dataset.severity = e.severity;
      li.tabIndex = 0;

      const tags = [];
      if (e.has_clip || e.clip_state === "ready") tags.push(`<span class="tag tag--clip">Clip</span>`);
      if (e.simulated) tags.push(`<span class="tag tag--sim">Simulated</span>`);
      if (e.acknowledged) tags.push(`<span class="tag tag--ack">Seen</span>`);

      li.innerHTML = `
        <time class="entry__time">${escape(e.clock || fmt.clock(e.ts))}</time>
        <div class="entry__body">
          <p class="entry__title">${escape(e.title || e.type)}</p>
          <p class="entry__where">${escape(e.camera)}${
            e.zone_name ? ` · ${escape(e.zone_name)}` : ""}</p>
          <p class="entry__text">${escape(e.message)}</p>
          ${tags.length ? `<div class="entry__tags">${tags.join("")}</div>` : ""}
        </div>`;
      li.addEventListener("click", () => openEvent(e.id));
      li.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openEvent(e.id); }
      });
      list.append(li);
    });
  }

  /* =======================================================================
     Incident detail
     ===================================================================== */

  async function openEvent(id) {
    const dialog = $("#event-dialog");
    const body = $("#event-detail");
    body.innerHTML = `<p class="empty">Opening incident ${id}…</p>`;
    dialog.showModal();

    let e;
    try { e = await api.get(`/api/events/${id}`); }
    catch (err) {
      body.innerHTML = `<p class="empty">Could not open that incident: ${escape(String(err))}</p>`;
      return;
    }

    $("#event-title").textContent = `${e.title} · ${e.camera}`;

    const measures = Object.entries(e.meta || {})
      .filter(([, v]) => ["number", "string", "boolean"].includes(typeof v))
      .map(([k, v]) => `<b>${escape(k.replace(/_/g, " "))}</b>  ${escape(String(v))}`)
      .join("\n");

    const clipBlock = e.clip
      ? `<video controls preload="metadata" playsinline
                src="${e.clip}?token=${encodeURIComponent(state.token)}"></video>
         <p class="event-detail__cap">Incident clip · 10s before, 5s after · the
            only raw footage this system holds of this moment</p>`
      : `<p class="event-detail__cap">${
            e.clip_state === "pending" ? "Clip is still being written…"
          : e.clip_state === "merged" ? "Folded into the clip for the first alert of this incident."
          : "No clip was saved for this kind of event."}</p>`;

    body.innerHTML = `
      <div class="event-detail__media">
        ${e.snapshot
          ? `<img src="${e.snapshot}" alt="Anonymous skeleton render of the incident">
             <p class="event-detail__cap">Skeleton snapshot · what the alert carried</p>`
          : `<p class="event-detail__cap">No snapshot was captured.</p>`}
        ${clipBlock}
      </div>
      <div class="event-detail__facts">
        <dl>
          ${row("Reference", `ARJUN-${String(e.id).padStart(5, "0")}`)}
          ${row("Kind", e.title)}
          ${row("Severity", e.severity.toUpperCase())}
          ${row("Camera", e.camera + (e.simulated ? " (simulated)" : ""))}
          ${row("Zone", e.zone_name || "not inside a defined zone")}
          ${row("Time", fmt.when(e.ts))}
        </dl>
        <p class="book__note">${escape(e.explanation || "")}</p>
        <p class="entry__text" style="-webkit-line-clamp:none;line-clamp:none;font-size:14px">
          ${escape(e.message)}</p>
        ${measures ? `<h3 class="book__head" style="margin-top:14px">What the rule measured</h3>
                      <pre class="measure">${measures}</pre>` : ""}
        <div class="event-actions">
          <button class="solid" id="btn-report">Write the report</button>
          <a class="ghost" href="/api/report/${e.id}/download" download
             style="text-decoration:none">Download as Markdown</a>
          <button class="ghost" id="btn-ack">${
            e.acknowledged ? "Mark unseen" : "Mark as seen"}</button>
        </div>
        <div id="report-slot"></div>
      </div>`;

    $("#btn-ack").addEventListener("click", async () => {
      await api.send(`/api/events/${e.id}/ack`, { acknowledged: !e.acknowledged });
      e.acknowledged = !e.acknowledged;
      $("#btn-ack").textContent = e.acknowledged ? "Mark unseen" : "Mark as seen";
      const found = state.events.find((x) => x.id === e.id);
      if (found) { found.acknowledged = e.acknowledged ? 1 : 0; renderTimeline(); }
    });

    $("#btn-report").addEventListener("click", async () => {
      const slot = $("#report-slot");
      const btn = $("#btn-report");
      btn.disabled = true;
      btn.textContent = "Writing…";
      slot.innerHTML = "";
      try {
        const res = await api.send(`/api/report/${e.id}`, {});
        slot.innerHTML = `<div class="report">${escape(res.answer)}</div>
          <p class="ask__meta">${escape(res.engine === "claude"
            ? `written by ${res.model}` : "written locally from the event record")}
          · ${res.events_used} events · ${res.elapsed_ms} ms</p>`;
      } catch (err) {
        slot.innerHTML = `<p class="empty">${escape(String(err))}</p>`;
      }
      btn.disabled = false;
      btn.textContent = "Write the report";
    });
  }

  const row = (k, v) =>
    `<div class="book__row"><dt>${escape(k)}</dt><dd>${escape(String(v))}</dd></div>`;

  /* =======================================================================
     Ask
     ===================================================================== */

  async function ask(question) {
    question = (question || "").trim();
    if (!question) return;
    const box = $("#ask-answer");
    box.hidden = false;
    box.textContent = "Reading the event ledger…";
    try {
      const res = await api.send("/api/ask", { question });
      box.innerHTML = `${escape(res.answer)}
        <p class="ask__meta">${escape(res.engine === "claude"
          ? `${res.model} · ${res.events_used} events`
          : `offline search · ${res.events_used} events`)}
        · ${res.elapsed_ms} ms${res.note ? ` · ${escape(res.note)}` : ""}</p>`;
    } catch (err) {
      box.textContent = `That did not work: ${err}`;
    }
  }

  /* =======================================================================
     Privacy ledger
     ===================================================================== */

  async function refreshPrivacy() {
    let data;
    try { data = await api.get("/api/privacy"); } catch { return; }

    const s = data.storage, db = data.database;
    $("#ledger-hours").textContent = s.continuous_hours;

    const memory = data.buffers.reduce((a, b) => a + (b.megabytes || 0), 0);
    const held = [
      ["Incident clips on disk", `${s.incident_clips}`,
       `Each one is ten seconds before an alert and five after. Nothing else is ever written.`],
      ["Space they take", fmt.bytes(s.clip_bytes), `Deleted automatically after ${s.retention_days} days.`],
      ["Skeleton snapshots", `${s.skeleton_snapshots}`,
       `Line drawings. Safe to send to a phone, which is exactly what the alerts do.`],
      ["Seconds of video in memory", `${data.buffer_seconds}`,
       `Per camera, ${memory.toFixed(1)} MB in all, overwritten continuously. Cut the power and it is gone.`],
      ["Incident records", `${db.events}`,
       `A time, a camera, a rule and a track number. ${fmt.bytes(db.megabytes * 1e6)} in total.`],
    ];
    renderBook($("#ledger-held"), held);

    const refused = (data.guarantees || [])
      .filter((g) => g.value === "0")
      .map((g) => [g.claim, g.value, g.why]);
    refused.push(["Face recognition models loaded", "0",
      "There is no face recognition anywhere in the system. The face mesh used by exam mode returns an angle, and the angle is all that is kept."]);
    renderBook($("#ledger-refused"), refused);

    renderAudit(data.recent_audit || []);
  }

  function renderBook(target, rows) {
    target.innerHTML = rows.map(([label, value, why]) => `
      <div class="book__row">
        <dt>${escape(label)}</dt>
        <dd>${escape(value)}</dd>
        <p class="book__why">${escape(why)}</p>
      </div>`).join("");
  }

  function renderAudit(rows) {
    const list = $("#audit-list");
    if (!rows.length) {
      list.innerHTML = `<li class="empty">No one has opened a raw feed.</li>`;
      return;
    }
    list.innerHTML = rows.map((a) => `
      <li class="${a.ok ? "" : "denied"}">
        <time>${escape(a.clock)}</time>
        <span class="what">${escape(a.action.replace(/_/g, " "))}</span>
        <span class="detail">${escape([a.camera, a.detail].filter(Boolean).join(" · "))}</span>
      </li>`).join("");
  }

  /* =======================================================================
     Analytics
     ===================================================================== */

  async function refreshAnalytics() {
    const hours = Number($("#analytics-range").value || 12);
    let data;
    try { data = await api.get(`/api/analytics?hours=${hours}`); } catch { return; }

    const ink = "#8a8375";
    const grid = "rgba(255,255,255,0.05)";
    Chart.defaults.color = ink;
    Chart.defaults.font.family = "IBM Plex Sans, system-ui, sans-serif";
    Chart.defaults.font.size = 11;

    // ---- by kind ------------------------------------------------------ #
    const palette = ["#d6a04a", "#c2483c", "#5e8c78", "#6782a8", "#d08a3e",
                     "#9a7bb0", "#b8b0a0", "#7d6a4f"];
    const types = data.by_type.slice(0, 8);
    const labels = types.map((t) => data.titles[t.type] || t.type);
    const counts = types.map((t) => t.n);

    if (state.charts.types) state.charts.types.destroy();
    state.charts.types = new Chart($("#chart-types"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: counts,
          backgroundColor: labels.map((_, i) => palette[i % palette.length]),
          borderRadius: 2,
          borderSkipped: false,
          barThickness: 14,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        // A fixed aspect rather than maintainAspectRatio:false — the figure
        // around the canvas has no set height, so "false" let both charts
        // grow to fill the column.
        aspectRatio: 1.85,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: grid }, border: { display: false },
               ticks: { precision: 0 } },
          y: { grid: { display: false }, border: { display: false } },
        },
      },
    });

    // ---- crowd over time ---------------------------------------------- #
    const byZone = new Map();
    (data.crowd || []).forEach((p) => {
      if (!byZone.has(p.zone)) byZone.set(p.zone, []);
      byZone.get(p.zone).push({ x: p.ts * 1000, y: p.count });
    });

    if (state.charts.crowd) state.charts.crowd.destroy();
    state.charts.crowd = new Chart($("#chart-crowd"), {
      type: "line",
      data: {
        datasets: Array.from(byZone.entries()).map(([zone, points], i) => ({
          label: zone,
          data: points,
          borderColor: palette[i % palette.length],
          backgroundColor: palette[i % palette.length] + "22",
          borderWidth: 1.6,
          tension: 0.32,
          pointRadius: 0,
          fill: true,
        })),
      },
      options: {
        responsive: true,
        aspectRatio: 1.85,
        plugins: {
          legend: { display: byZone.size > 0, labels: { boxWidth: 9, boxHeight: 9 } },
          tooltip: { intersect: false, mode: "index" },
        },
        scales: {
          x: {
            type: "linear",
            grid: { color: grid },
            border: { display: false },
            ticks: {
              maxTicksLimit: 6,
              callback: (v) => new Date(v).toLocaleTimeString([], {
                hour: "2-digit", minute: "2-digit", hour12: false }),
            },
          },
          y: { beginAtZero: true, grid: { color: grid },
               border: { display: false }, ticks: { precision: 0 } },
        },
      },
    });

    // ---- occupancy ------------------------------------------------------ #
    const occ = $("#occupancy");
    occ.innerHTML = (data.occupancy || []).map((o) => `
      <div class="occupancy__item">
        <div class="occupancy__count">${o.inside}</div>
        <div class="occupancy__label">inside · ${escape(o.camera)}</div>
      </div>`).join("")
      || `<p class="empty" style="padding:0">Draw a counting line on a camera to
          count people in and out.</p>`;
  }

  /* =======================================================================
     Rules
     ===================================================================== */

  async function refreshRules() {
    let data;
    try { data = await api.get("/api/detectors"); } catch { return; }

    const merged = new Map();
    Object.values(data).forEach((rules) => {
      rules.forEach((r) => {
        const prev = merged.get(r.name);
        if (prev) { prev.fired += r.fired; prev.errors += r.errors; }
        else merged.set(r.name, { ...r });
      });
    });

    $("#rules-list").innerHTML = Array.from(merged.values()).map((r) => `
      <article class="rule-card" data-muted="${r.muted}">
        <h3>${escape(r.title)}
          <span class="rule-card__count">${r.fired}</span></h3>
        <p>${escape(r.explanation || "")}</p>
        ${r.last_error ? `<p style="color:var(--cinnabar)">${escape(r.last_error)}</p>` : ""}
      </article>`).join("");
  }

  /* =======================================================================
     Zone editor
     ===================================================================== */

  const editor = {
    camera: null, points: [], zones: [], image: null,
    canvas: null, ctx: null,
  };

  async function openZoneEditor(camera) {
    editor.camera = camera;
    editor.points = [];
    editor.canvas = $("#zone-canvas");
    editor.ctx = editor.canvas.getContext("2d");
    $("#zone-camera").textContent = camera;
    $("#zone-name").value = "";
    $("#zone-dialog").showModal();
    updateTypeHint();

    const img = new Image();
    img.onload = () => {
      editor.image = img;
      editor.canvas.width = img.naturalWidth;
      editor.canvas.height = img.naturalHeight;
      drawEditor();
    };
    img.src = `/api/frame/${encodeURIComponent(camera)}?t=${Date.now()}`;

    await loadZones();
  }

  async function loadZones() {
    const data = await api.get(`/api/zones?camera=${encodeURIComponent(editor.camera)}`);
    editor.zones = data.zones;
    renderZoneList();
    drawEditor();
  }

  function renderZoneList() {
    const list = $("#zone-list");
    if (!editor.zones.length) {
      list.innerHTML = `<li class="empty" style="padding:8px 0">No zones here yet.</li>`;
      return;
    }
    list.innerHTML = "";
    editor.zones.forEach((z) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <span class="swatch" style="background:${z.colour}"></span>
        <span class="nm">${escape(z.name)}</span>
        <span class="kind">${escape(z.type)}</span>
        <button type="button" title="Delete this zone">×</button>`;
      $("button", li).addEventListener("click", async () => {
        await api.send(`/api/zones/${z.id}`, undefined, "DELETE");
        await loadZones();
      });
      list.append(li);
    });
  }

  function drawEditor() {
    const { ctx, canvas, image } = editor;
    if (!ctx || !canvas.width) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (image) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    editor.zones.forEach((z) => drawShape(
      z.points.map(([x, y]) => [x * canvas.width, y * canvas.height]),
      z.colour, z.type === "line", z.name));

    if (editor.points.length) {
      const type = $("#zone-type").value;
      drawShape(editor.points, "#ecc274", type === "line", null, true);
    }
  }

  function drawShape(pts, colour, isLine, label, live = false) {
    const { ctx } = editor;
    if (pts.length < 1) return;
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = colour;
    ctx.setLineDash(live ? [6, 4] : []);

    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (!isLine && pts.length > 2 && !live) ctx.closePath();
    ctx.stroke();

    if (!isLine && pts.length > 2) {
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    ctx.fillStyle = colour;
    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    if (label) {
      ctx.font = "13px 'IBM Plex Sans', sans-serif";
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
      const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      const w = ctx.measureText(label).width + 10;
      ctx.fillStyle = "rgba(10,10,11,0.76)";
      ctx.fillRect(cx - w / 2, cy - 10, w, 20);
      ctx.fillStyle = colour;
      ctx.textAlign = "center";
      ctx.fillText(label, cx, cy + 4);
    }
    ctx.restore();
  }

  function canvasPoint(ev) {
    const rect = editor.canvas.getBoundingClientRect();
    const scaleX = editor.canvas.width / rect.width;
    const scaleY = editor.canvas.height / rect.height;
    return [(ev.clientX - rect.left) * scaleX, (ev.clientY - rect.top) * scaleY];
  }

  function wireEditor() {
    const canvas = $("#zone-canvas");

    canvas.addEventListener("click", (ev) => {
      const [x, y] = canvasPoint(ev);
      const isLine = $("#zone-type").value === "line";

      // Clicking the first point again closes the shape.
      if (!isLine && editor.points.length > 2) {
        const [fx, fy] = editor.points[0];
        if (Math.hypot(fx - x, fy - y) < 14) return finishShape();
      }
      editor.points.push([x, y]);
      if (isLine && editor.points.length === 2) return finishShape();
      drawEditor();
      $("#zone-save").disabled = editor.points.length < (isLine ? 2 : 3);
    });

    document.addEventListener("keydown", (ev) => {
      if (!$("#zone-dialog").open) return;
      if (ev.key === "Enter") { ev.preventDefault(); finishShape(); }
      if (ev.key === "Backspace") { ev.preventDefault(); undoPoint(); }
    });

    $("#zone-undo").addEventListener("click", undoPoint);
    $("#zone-clear").addEventListener("click", () => {
      editor.points = [];
      $("#zone-save").disabled = true;
      drawEditor();
    });
    $("#zone-save").addEventListener("click", finishShape);
    $("#zone-type").addEventListener("change", () => {
      editor.points = [];
      $("#zone-save").disabled = true;
      updateTypeHint();
      drawEditor();
    });
  }

  function undoPoint() {
    editor.points.pop();
    const isLine = $("#zone-type").value === "line";
    $("#zone-save").disabled = editor.points.length < (isLine ? 2 : 3);
    drawEditor();
  }

  const TYPE_HINTS = {
    restricted: "Alerts the moment somebody's feet enter this area.",
    loiter: "Alerts when somebody stays inside longer than the loitering time.",
    crowd: "Counts people standing inside, and warns when it gets busy.",
    exit: "Alerts when anything is left here, or somebody stands rather than walks through.",
    desk: "Exam mode watches this desk for a visible phone or a sustained head turn.",
    line: "Two points. People crossing are counted in and out, giving a live figure for how many are inside.",
  };

  function updateTypeHint() {
    const type = $("#zone-type").value;
    $("#zone-type-hint").textContent = TYPE_HINTS[type] || "";
    $("#zone-hint").textContent = type === "line"
      ? "Click the two ends of the line. People crossing it are counted."
      : "Click to place corners. Click the first corner again, or press Enter, to close the shape.";
  }

  async function finishShape() {
    const type = $("#zone-type").value;
    const need = type === "line" ? 2 : 3;
    if (editor.points.length < need) return;

    const name = $("#zone-name").value.trim()
      || `${type[0].toUpperCase()}${type.slice(1)} ${editor.zones.length + 1}`;
    const { canvas } = editor;
    const points = editor.points
      .slice(0, type === "line" ? 2 : undefined)
      .map(([x, y]) => [x / canvas.width, y / canvas.height]);

    try {
      await api.send("/api/zones", { camera: editor.camera, name, type, points });
      editor.points = [];
      $("#zone-name").value = "";
      $("#zone-save").disabled = true;
      await loadZones();
      toast({ title: "Zone saved", message: `${name} is live on ${editor.camera}.`,
              severity: "info" });
    } catch (err) {
      toast({ title: "Could not save that zone", message: String(err),
              severity: "warning" });
    }
  }

  /* =======================================================================
     Toasts and sound
     ===================================================================== */

  function toast({ title, message, severity = "info", onClick }) {
    const host = $("#toasts");
    const el = document.createElement("div");
    el.className = "toast";
    el.dataset.severity = severity;
    el.innerHTML = `<p class="toast__title">${escape(title)}</p>
                    <p class="toast__text">${escape(message || "")}</p>`;
    el.addEventListener("click", () => { onClick?.(); dismiss(el); });
    host.append(el);
    setTimeout(() => dismiss(el), severity === "critical" ? 10000 : 5500);

    // Trim the stack to the newest four. This counts only the toasts that are
    // not already leaving, because dismiss() fades an element out and removes
    // it 320ms later - a loop that waited for children.length to fall would
    // spin forever on the main thread, which is exactly what it did.
    const live = () => Array.from(host.children)
                            .filter((n) => !n.classList.contains("is-going"));
    let standing = live();
    while (standing.length > 3) dismiss(standing.shift());
  }

  function dismiss(el) {
    if (!el || el.classList.contains("is-going")) return;
    el.classList.add("is-going");
    setTimeout(() => el.remove(), 320);
  }

  let audio;
  function chime(urgent) {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === "suspended") audio.resume();
      const now = audio.currentTime;
      // A minor third for a warning, a falling fifth for something critical:
      // enough to tell them apart across a room without being an alarm bell.
      const notes = urgent ? [784, 523] : [659, 784];
      notes.forEach((freq, i) => {
        const osc = audio.createOscillator();
        const gain = audio.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        const at = now + i * 0.17;
        gain.gain.setValueAtTime(0, at);
        gain.gain.linearRampToValueAtTime(urgent ? 0.1 : 0.055, at + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.42);
        osc.connect(gain).connect(audio.destination);
        osc.start(at);
        osc.stop(at + 0.45);
      });
    } catch { /* a browser that will not make a sound is not an error */ }
  }

  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
  }

  /* ===================================================================== */

  wireEditor();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();

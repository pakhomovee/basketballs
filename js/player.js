/** Standalone annotated-video viewer (read-only demo). */

import {
	drawAnnotations,
	resolveFrame,
	annotationFrameStep,
	playerColor,
	DEFAULT_TOGGLES
} from './draw.js';

const TEAM_COLORS = ['#ff6b2b', '#3b82f6'];
const teamColor = (t) => TEAM_COLORS[t] ?? TEAM_COLORS[0];

// --- DOM handles -----------------------------------------------------------
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const stage = document.getElementById('stage');
const msg = document.getElementById('msg');
const titleEl = document.getElementById('title');
const metaEl = document.getElementById('meta');
const noteEl = document.getElementById('note');
const playBtn = document.getElementById('playBtn');
const timeEl = document.getElementById('time');
const seek = document.getElementById('seek');
const frameCountEl = document.getElementById('framecount');
const ratesEl = document.getElementById('rates');
const togglesEl = document.getElementById('toggles');
const possessionEl = document.getElementById('possession');
const possOwnerEl = document.getElementById('possOwner');
const statPassesEl = document.getElementById('statPasses');
const statShotsEl = document.getElementById('statShots');
const statMakesEl = document.getElementById('statMakes');
const eventsEl = document.getElementById('events');

const ctx = canvas.getContext('2d');

// Event rows, kept for per-frame active-highlighting.
let eventRows = [];

// --- state -----------------------------------------------------------------
let annotations = null;
let toggles = { ...DEFAULT_TOGGLES };
let fps = 30;
let step = 1;
let currentFrame = 0;
let displayW = 0;
let displayH = 0;
let rvfcHandle = -1;
let rafHandle = 0;

const PLAY_ICON = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>';
const PAUSE_ICON = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

// --- helpers ---------------------------------------------------------------
function qparam(name) {
	return new URLSearchParams(location.search).get(name);
}

function formatTime(sec) {
	if (!Number.isFinite(sec)) sec = 0;
	const m = Math.floor(sec / 60);
	const s = Math.floor(sec % 60);
	return `${m}:${s.toString().padStart(2, '0')}`;
}

function timeToFrame(t) {
	return Math.floor(t * fps + 1e-9);
}

function showMessage(html) {
	msg.innerHTML = html;
	msg.hidden = false;
}

// --- rendering -------------------------------------------------------------
function sizeCanvas() {
	// Video displays at 100% container width, intrinsic aspect ratio.
	const w = video.clientWidth;
	const h = video.clientHeight;
	if (w === 0 || h === 0) return;
	displayW = w;
	displayH = h;
	canvas.width = w;
	canvas.height = h;
	canvas.style.width = w + 'px';
	canvas.style.height = h + 'px';
	render();
}

function render() {
	if (!annotations || displayW === 0) return;
	ctx.clearRect(0, 0, displayW, displayH);
	const ef = resolveFrame(annotations, currentFrame);
	const fd = annotations.frames[String(ef)];
	if (!fd) return;
	drawAnnotations(
		ctx,
		fd,
		toggles,
		displayW / annotations.metadata.width,
		displayH / annotations.metadata.height
	);
}

// `mediaTime`, when provided, is the exact presentation timestamp of the frame
// the browser just painted (from requestVideoFrameCallback). Prefer it over
// `video.currentTime`: inside a rVFC callback currentTime is the coarse
// main-thread clock and often still reads the *previous* frame's time, which
// makes the overlay trail the video by up to a frame. mediaTime keeps the
// annotations locked to the frame actually on screen.
function syncFrame(mediaTime) {
	// Guard with typeof, not `??`: event listeners (e.g. 'seeked') can call this
	// with an Event object, which would slip past a null-check and poison
	// timeToFrame() with NaN.
	const frameTime = typeof mediaTime === 'number' ? mediaTime : video.currentTime;
	const nf = timeToFrame(frameTime);
	if (nf !== currentFrame) {
		currentFrame = nf;
		frameCountEl.textContent = `F${currentFrame}`;
		render();
		updateGameState(currentFrame);
	}
	const t = video.currentTime;
	const dur = video.duration || 0;
	timeEl.textContent = `${formatTime(t)} / ${formatTime(dur)}`;
	if (dur > 0) seek.value = String(Math.round((t / dur) * 1000));
}

function startFrameLoop() {
	if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
		const onFrame = (_now, metadata) => {
			syncFrame(metadata.mediaTime);
			rvfcHandle = video.requestVideoFrameCallback(onFrame);
		};
		rvfcHandle = video.requestVideoFrameCallback(onFrame);
	} else {
		const frame = () => {
			if (!video.paused) syncFrame();
			rafHandle = requestAnimationFrame(frame);
		};
		rafHandle = requestAnimationFrame(frame);
	}
}

// --- controls --------------------------------------------------------------
function togglePlay() {
	if (video.paused) video.play();
	else video.pause();
}

function stepFrame(delta) {
	if (!fps) return;
	video.pause();
	video.currentTime = Math.max(0, video.currentTime + (delta / fps));
}

function buildRates() {
	for (const rate of [0.25, 0.5, 1, 2]) {
		const b = document.createElement('button');
		b.className = 'rate' + (rate === 1 ? ' active' : '');
		b.textContent = `${rate}x`;
		b.onclick = () => {
			video.playbackRate = rate;
			ratesEl.querySelectorAll('.rate').forEach((el) => el.classList.remove('active'));
			b.classList.add('active');
		};
		ratesEl.appendChild(b);
	}
}

const TOGGLE_ITEMS = [
	{ key: 'bboxes', label: 'Bboxes', icon: '▢' },
	{ key: 'masks', label: 'Masks', icon: '◉' },
	{ key: 'skeletons', label: 'Skeletons', icon: '🦴' },
	{ key: 'ball', label: 'Ball', icon: '🏀' },
	{ key: 'trackNumbers', label: 'Track #', icon: '#' },
	{ key: 'rawJerseyNumbers', label: 'Raw #', icon: '⌗' },
	{ key: 'playerIds', label: 'IDs', icon: 'ID' }
];

function isDisabled(key) {
	return key === 'playerIds' && !toggles.bboxes;
}

function buildToggles() {
	togglesEl.replaceChildren();
	for (const item of TOGGLE_ITEMS) {
		const b = document.createElement('button');
		const disabled = isDisabled(item.key);
		b.className = 'toggle' + (toggles[item.key] ? ' on' : '') + (disabled ? ' disabled' : '');
		b.innerHTML = `<span class="ic">${item.icon}</span>${item.label}`;
		b.onclick = () => {
			if (isDisabled(item.key)) return;
			toggles = { ...toggles, [item.key]: !toggles[item.key] };
			// playerIds requires bboxes — turn it off when bboxes are hidden
			if (item.key === 'bboxes' && !toggles.bboxes) toggles.playerIds = false;
			buildToggles();
			render();
		};
		togglesEl.appendChild(b);
	}

	// color-mode segmented control
	const seg = document.createElement('div');
	seg.className = 'segmented';
	for (const mode of ['team', 'id']) {
		const b = document.createElement('button');
		b.className = toggles.colorMode === mode ? 'active' : '';
		b.textContent = mode === 'team' ? 'Team' : 'ID';
		b.onclick = () => {
			toggles = { ...toggles, colorMode: mode };
			buildToggles();
			render();
			updatePossession(currentFrame);
		};
		seg.appendChild(b);
	}
	togglesEl.appendChild(seg);
}

// --- game-state panel ------------------------------------------------------
function ownerLabel(p) {
	const parts = [];
	if (p.track_number != null) parts.push(`#${p.track_number}`);
	parts.push(`ID ${p.player_id}`);
	if (p.team_id != null) parts.push(`Team ${p.team_id + 1}`);
	return parts.join(' · ');
}

function updatePossession(frame) {
	if (!annotations) return;
	const fd = annotations.frames[String(resolveFrame(annotations, frame))];
	const owner = fd ? fd.players.find((p) => p.is_possession) : null;
	if (owner) {
		possessionEl.classList.add('active');
		possOwnerEl.classList.remove('empty');
		possOwnerEl.innerHTML = '';
		const sw = document.createElement('span');
		sw.className = 'swatch';
		sw.style.background = playerColor(owner, toggles.colorMode);
		possOwnerEl.append(sw, document.createTextNode(ownerLabel(owner)));
	} else {
		possessionEl.classList.remove('active');
		possOwnerEl.classList.add('empty');
		possOwnerEl.textContent = 'No possession';
	}
}

function playerTag(color, text) {
	const s = document.createElement('span');
	s.className = 'event-detail';
	const sw = document.createElement('span');
	sw.className = 'swatch';
	sw.style.background = color;
	s.append(sw, document.createTextNode(text));
	return s;
}

function buildGameState() {
	const passes = annotations.pass_events ?? [];
	const shots = annotations.shot_events ?? [];
	statPassesEl.textContent = String(passes.length);
	statShotsEl.textContent = String(shots.length);
	statMakesEl.textContent = String(shots.filter((s) => s.is_make).length);

	const rows = [
		...passes.map((p) => ({ kind: 'pass', data: p, sort: p.frame_start })),
		...shots.map((s) => ({ kind: 'shot', data: s, sort: s.frame_start }))
	].sort((a, b) => a.sort - b.sort || (a.kind === 'pass' ? -1 : 1));

	eventsEl.replaceChildren();
	eventRows = [];

	if (rows.length === 0) {
		const p = document.createElement('div');
		p.className = 'empty';
		p.textContent = 'No events in this clip';
		eventsEl.appendChild(p);
		return;
	}

	for (const row of rows) {
		const btn = document.createElement('button');
		btn.className = 'event' + (row.kind === 'shot' && row.data.is_make ? ' make' : '');

		const top = document.createElement('div');
		top.className = 'event-top';

		if (row.kind === 'pass') {
			const e = row.data;
			top.appendChild(tag('pass', 'Pass'));
			top.appendChild(timeChip(formatTime(e.timestamp_sec)));
			top.appendChild(frames(e.frame_start, e.frame_end));
			btn.appendChild(top);

			const detail = document.createElement('div');
			detail.className = 'event-detail';
			const c = teamColor(e.team_id);
			const from = document.createElement('span');
			from.style.color = c;
			from.textContent =
				(e.from_track_number != null ? `#${e.from_track_number} ` : '') + `ID ${e.from_player_id}`;
			const arrow = document.createElement('span');
			arrow.className = 'arrow';
			arrow.textContent = '→';
			const to = document.createElement('span');
			to.style.color = c;
			to.textContent =
				(e.to_track_number != null ? `#${e.to_track_number} ` : '') + `ID ${e.to_player_id}`;
			const sw = document.createElement('span');
			sw.className = 'swatch';
			sw.style.background = c;
			detail.append(sw, from, arrow, to);
			btn.appendChild(detail);

			btn.onclick = () => seekSeconds((e.frame_start + 0.5) / fps);
			eventRows.push({ el: btn, start: e.frame_start, end: e.frame_end });
		} else {
			const e = row.data;
			top.appendChild(tag(e.is_make ? 'make' : 'shot', 'Shot'));
			top.appendChild(timeChip(`${formatTime(e.timestamp_start_sec)}–${formatTime(e.timestamp_end_sec)}`));
			top.appendChild(frames(e.frame_start, e.frame_end));
			if (e.is_make) top.appendChild(tag('make', 'Make'));
			btn.appendChild(top);

			if (e.shooter_player_id != null) {
				const c = e.shooter_team_id != null ? teamColor(e.shooter_team_id) : 'var(--text-muted)';
				const label =
					(e.shooter_track_number != null ? `#${e.shooter_track_number} ` : '') +
					`ID ${e.shooter_player_id}`;
				btn.appendChild(playerTag(c, label));
			}

			btn.onclick = () => seekSeconds(e.timestamp_start_sec);
			eventRows.push({ el: btn, start: e.frame_start, end: e.frame_end });
		}

		eventsEl.appendChild(btn);
	}
}

function tag(cls, text) {
	const s = document.createElement('span');
	s.className = `tag ${cls}`;
	s.textContent = text;
	return s;
}
function timeChip(text) {
	const s = document.createElement('span');
	s.className = 'event-time';
	s.textContent = text;
	return s;
}
function frames(a, b) {
	const s = document.createElement('span');
	s.className = 'event-frames';
	s.textContent = `f${a}–${b}`;
	return s;
}

function updateActiveEvent(frame) {
	for (const r of eventRows) {
		const active = frame >= r.start && frame <= r.end;
		r.el.classList.toggle('active', active);
	}
}

function seekSeconds(t) {
	if (Number.isFinite(t)) video.currentTime = Math.max(0, t);
}

function updateGameState(frame) {
	updatePossession(frame);
	updateActiveEvent(frame);
}

// --- init ------------------------------------------------------------------
async function main() {
	buildRates();
	buildToggles();

	const id = qparam('v');
	let entry = null;
	try {
		const { videos } = await (await fetch('data/manifest.json')).json();
		entry = videos.find((v) => v.id === id) ?? null;
	} catch (e) {
		showMessage(`Could not load clip list.<br /><span style="opacity:.6">${e.message}</span>`);
		return;
	}

	if (!entry) {
		titleEl.textContent = 'Clip not found';
		showMessage('This clip does not exist. <a href="index.html">Back to all clips</a>.');
		return;
	}

	titleEl.textContent = entry.title;
	document.title = `${entry.title} — Basketballs`;
	noteEl.textContent = entry.description;

	if (entry.available === false) {
		showMessage('Annotations for this clip are not available yet.');
		return;
	}

	// Load video + annotations in parallel.
	video.src = entry.video;
	video.addEventListener('error', () =>
		showMessage('Could not load video.<br /><span style="opacity:.6">File missing or unsupported format.</span>')
	);

	try {
		annotations = await (await fetch(entry.annotations)).json();
	} catch (e) {
		showMessage(`Could not load annotations.<br /><span style="opacity:.6">${e.message}</span>`);
		return;
	}

	fps = annotations.metadata.fps ?? 30;
	step = annotationFrameStep(annotations);
	metaEl.textContent =
		`${annotations.metadata.width}×${annotations.metadata.height} · ${fps} fps · ` +
		`${annotations.metadata.total_frames} frames`;

	buildGameState();
	updateGameState(0);

	// wire video events
	video.addEventListener('loadedmetadata', () => {
		sizeCanvas();
		syncFrame();
	});
	video.addEventListener('seeked', () => syncFrame());
	video.addEventListener('timeupdate', () => {
		if (video.paused) syncFrame();
	});
	video.addEventListener('play', () => (playBtn.innerHTML = PAUSE_ICON));
	video.addEventListener('pause', () => (playBtn.innerHTML = PLAY_ICON));

	new ResizeObserver(sizeCanvas).observe(stage);
	startFrameLoop();

	// wire controls
	playBtn.onclick = togglePlay;
	document.getElementById('prevFrame').onclick = () => stepFrame(-1);
	document.getElementById('nextFrame').onclick = () => stepFrame(1);
	seek.addEventListener('input', () => {
		const dur = video.duration || 0;
		if (dur > 0) video.currentTime = (Number(seek.value) / 1000) * dur;
	});
	window.addEventListener('keydown', (e) => {
		const tag = e.target?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA') return;
		if (e.key === 'ArrowLeft') {
			e.preventDefault();
			stepFrame(-1);
		} else if (e.key === 'ArrowRight') {
			e.preventDefault();
			stepFrame(1);
		} else if (e.key === ' ') {
			e.preventDefault();
			togglePlay();
		}
	});
}

main();

<script lang="ts">
	import { onMount } from 'svelte';
	import type { AnnotationData, ToggleState, FrameAnnotation } from '$lib/types';
	import AnnotationCanvas from './AnnotationCanvas.svelte';
	import CourtMinimap from './CourtMinimap.svelte';
	import SeekBarTimeline from './SeekBarTimeline.svelte';
	import { Muxer, ArrayBufferTarget } from 'mp4-muxer';
	import { annotationFrameStep, drawAnnotations, resolveFrame } from '$lib/draw';

	interface Props {
		src: string;
		annotations: AnnotationData | null;
		toggles: ToggleState;
		/** Bindable: current time in seconds */
		currentTime?: number;
		/** Bindable: current frame index */
		currentFrame?: number;
		/** Bindable output: accumulated court-position trails for minimap */
		courtTrails?: Record<number, [number, number][]>;
		/** Bindable output: actual rendered player width */
		playerWidth?: number;
	}

	let {
		src,
		annotations,
		toggles,
		currentTime = $bindable(0),
		currentFrame = $bindable(0),
		courtTrails = $bindable({}),
		playerWidth = $bindable(0)
	}: Props = $props();

	let video: HTMLVideoElement;
	let container: HTMLDivElement;
	let playing = $state(false);
	let duration = $state(0);
	let displayWidth = $state(0);
	let displayHeight = $state(0);
	let playbackRate = $state(1);
	let videoError = $state(false);
	let animFrame: number;
	let rvfcHandle = -1; // requestVideoFrameCallback handle when supported
	const TRAIL_LENGTH = 30;

	// Minimap state exposed from CourtMinimap for export
	let minimapCanvas: HTMLCanvasElement | null = $state(null);
	let minimapDragRight = $state(16);
	let minimapDragBottom = $state(16);

	// Use the pipeline's target fps directly — this is what the logical frame
	// indices in the annotations are based on.  Deriving fps from
	// total_frames / browser_duration is fragile: the browser may report a
	// different duration (e.g. when the last N source frames are corrupt,
	// or the container has a non-zero start PTS).
	const fps = $derived(annotations?.metadata.fps ?? 30);

	const annotationStep = $derived.by(() => {
		if (!annotations) return 1;
		return annotationFrameStep(annotations);
	});

	const displayFps = $derived.by(() => fps / annotationStep);

	// Floor currentFrame to the nearest annotated frame (≤ currentFrame).
	// With duplicate-pair source video (30fps→60fps) and frame_step=2, annotation
	// keys are 0,2,4,... Raw video frames 0 and 1 both resolve to annotation "0",
	// frames 2 and 3 resolve to "2", etc. — no flickering and no look-ahead.
	const effectiveFrame = $derived.by(() => {
		if (!annotations) return currentFrame;
		return resolveFrame(annotations, currentFrame);
	});

	onMount(() => {
		const observer = new ResizeObserver(() => updateSize());
		observer.observe(container);
		startFrameLoop();

		function handleKeyDown(e: KeyboardEvent) {
			const tag = (e.target as HTMLElement)?.tagName;
			if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) {
				return;
			}
			if (e.key === 'ArrowLeft') {
				e.preventDefault();
				stepFrame(-1);
			} else if (e.key === 'ArrowRight') {
				e.preventDefault();
				stepFrame(1);
			}
		}
		window.addEventListener('keydown', handleKeyDown);

		return () => {
			observer.disconnect();
			if (rvfcHandle >= 0) (video as any).cancelVideoFrameCallback(rvfcHandle);
			else if (animFrame) cancelAnimationFrame(animFrame);
			window.removeEventListener('keydown', handleKeyDown);
		};
	});

	// Re-compute size when annotations arrive (aspect ratio may differ from 16/9 default)
	$effect(() => {
		if (annotations) updateSize();
	});

	function timeToFrame(t: number): number {
		return Math.floor(t * fps + 1e-9);
	}

	function syncFrame() {
		const t = video.currentTime;
		currentTime = t;
		const newFrame = timeToFrame(t);
		if (newFrame !== currentFrame) {
			currentFrame = newFrame;
			updateTrails(newFrame);
		}
	}

	function startFrameLoop() {
		if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
			function onVideoFrame() {
				syncFrame();
				rvfcHandle = (video as any).requestVideoFrameCallback(onVideoFrame);
			}
			rvfcHandle = (video as any).requestVideoFrameCallback(onVideoFrame);
		} else {
			function frame() {
				if (video && !video.paused) syncFrame();
				animFrame = requestAnimationFrame(frame);
			}
			animFrame = requestAnimationFrame(frame);
		}
	}

	function updateSize() {
		if (!container) return;
		const cW = container.clientWidth;
		const aspect = annotations ? annotations.metadata.width / annotations.metadata.height : 16 / 9;
		const maxH = Math.floor(window.innerHeight * 0.74);
		const heightFromWidth = Math.round(cW / aspect);
		if (heightFromWidth > maxH) {
			displayHeight = maxH;
			displayWidth = Math.round(maxH * aspect);
		} else {
			displayWidth = cW;
			displayHeight = heightFromWidth;
		}
		playerWidth = displayWidth;
	}

	function onLoadedMetadata() {
		duration = video.duration;
		videoError = false;
		updateSize();
		syncFrame();
	}

	function onTimeUpdate() {
		if (video.paused && !exporting) syncFrame();
	}

	function updateTrails(frame: number) {
		if (!annotations) return;
		const ef = resolveFrame(annotations, frame);
		const fd: FrameAnnotation | undefined = annotations.frames[String(ef)];
		if (!fd) return;
		const newTrails = { ...courtTrails };
		for (const p of fd.players) {
			if (p.player_id < 0 || !p.court_position) continue;
			const [cx, cy] = p.court_position;
			if (!newTrails[p.player_id]) newTrails[p.player_id] = [];
			newTrails[p.player_id].push([cx, cy]);
			if (newTrails[p.player_id].length > TRAIL_LENGTH) {
				newTrails[p.player_id] = newTrails[p.player_id].slice(-TRAIL_LENGTH);
			}
		}
		courtTrails = newTrails;
	}

	function stepFrame(delta: number) {
		if (!video || !fps) return;
		video.pause();
		playing = false;
		const stepSeconds = 1 / fps;
		const newTime = Math.max(0, video.currentTime + delta * stepSeconds);
		video.currentTime = newTime;
		courtTrails = {};
	}

	function togglePlay() {
		if (video.paused) {
			video.play();
			playing = true;
		} else {
			video.pause();
			playing = false;
		}
	}

	function setRate(rate: number) {
		playbackRate = rate;
		video.playbackRate = rate;
	}

	// Use actual video duration when known; fall back to annotations metadata
	const displayDuration = $derived(
		duration > 0
			? duration
			: annotations
				? annotations.metadata.total_frames / (annotations.metadata.fps || 30)
				: 0
	);

	function formatTime(sec: number): string {
		const m = Math.floor(sec / 60);
		const s = Math.floor(sec % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	export function seekTo(time: number) {
		if (video) {
			video.currentTime = time;
			courtTrails = {};
		}
	}

	let exporting = $state(false);
	let exportProgress = $state(0);

	export async function exportVideo() {
		if (!video || !annotations || exporting) return;

		exporting = true;
		exportProgress = 0;

		try {
			const natW = annotations.metadata.width;
			const natH = annotations.metadata.height;
			const totalFrames = Math.ceil(annotations.metadata.total_frames / annotationStep);
			const exportFps = displayFps;
			const frameDurationUs = Math.round(1_000_000 / exportFps);

			const offscreen = document.createElement('canvas');
			offscreen.width = natW;
			offscreen.height = natH;
			const offCtx = offscreen.getContext('2d')!;

			// A detached video element — never added to the DOM, so the visible
			// player is completely unaffected during the export loop.
			const exportVid = document.createElement('video');
			exportVid.src = src;
			exportVid.muted = true;
			exportVid.preload = 'auto';
			await new Promise<void>((resolve, reject) => {
				exportVid.onloadedmetadata = () => resolve();
				exportVid.onerror = () => reject(new Error('Failed to load video for export'));
			});

			function waitForSeek(el: HTMLVideoElement): Promise<void> {
				return new Promise<void>((resolve) => {
					el.addEventListener('seeked', () => resolve(), { once: true });
				});
			}

			// mp4-muxer + VideoEncoder: assigns exact timestamps regardless of wall-clock time
			const target = new ArrayBufferTarget();
			const muxer = new Muxer({
				target,
				video: { codec: 'avc', width: natW, height: natH, frameRate: exportFps },
				fastStart: 'in-memory'
			});

			const encoder = new VideoEncoder({
				output: (chunk, meta) => muxer.addVideoChunk(chunk, meta ?? undefined),
				error: (e) => console.error('VideoEncoder error', e)
			});
			encoder.configure({
				codec: 'avc1.640028',
				width: natW,
				height: natH,
				bitrate: 8_000_000,
				framerate: exportFps
			});

			const lastAnnotatedFrame = Math.max(...Object.keys(annotations.frames).map(Number));
			const exportTotal = Math.min(
				totalFrames,
				Math.floor(lastAnnotatedFrame / annotationStep) + 1
			);

			for (let f = 0; f < exportTotal; f++) {
				const frameIdx = f * annotationStep;
				const seekTime = Math.min(frameIdx / fps, exportVid.duration - 0.001);
				exportVid.currentTime = seekTime;
				await waitForSeek(exportVid);

				// Compose frame onto offscreen canvas
				offCtx.drawImage(exportVid, 0, 0, natW, natH);
				const fd = annotations.frames[String(resolveFrame(annotations, frameIdx))];
				if (fd) drawAnnotations(offCtx, fd, toggles, 1, 1);

				if (toggles.minimap && minimapCanvas) {
					const scale = natW / displayWidth;
					const scaledW = minimapCanvas.width * scale;
					const scaledH = minimapCanvas.height * scale;
					const mx = natW - minimapDragRight * scale - scaledW;
					const my = natH - minimapDragBottom * scale - scaledH;
					offCtx.drawImage(minimapCanvas, mx, my, scaledW, scaledH);
				}

				// Encode with exact timestamp — no wall-clock involvement
				const timestamp = f * frameDurationUs;
				const videoFrame = new VideoFrame(offscreen, { timestamp, duration: frameDurationUs });
				encoder.encode(videoFrame, { keyFrame: f % 30 === 0 });
				videoFrame.close();

				// Drain encoder queue periodically to avoid unbounded buffering
				if (encoder.encodeQueueSize > 10) {
					await new Promise<void>((r) => setTimeout(r, 0));
				}

				exportProgress = Math.round(((f + 1) / exportTotal) * 100);
			}

			await encoder.flush();
			muxer.finalize();

			// Free the detached video element
			exportVid.src = '';

			const blob = new Blob([target.buffer], { type: 'video/mp4' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `export-${annotations.metadata.video_name.replace(/\.[^.]+$/, '')}.mp4`;
			a.click();
			URL.revokeObjectURL(url);
		} finally {
			exporting = false;
			exportProgress = 0;
		}
	}
</script>

<!-- Measurement wrapper spans available width; inner player is constrained to video width -->
<div bind:this={container} class="w-full">
	<div class="flex flex-col gap-3" style="width: {displayWidth > 0 ? displayWidth + 'px' : '100%'}">
		<div
			class="relative bg-black rounded-xl overflow-hidden"
			style="width: {displayWidth}px; height: {displayHeight}px;"
		>
			<video
				bind:this={video}
				{src}
				onloadedmetadata={onLoadedMetadata}
				ontimeupdate={onTimeUpdate}
				onseeked={() => {
					if (!exporting) syncFrame();
				}}
				onplay={() => {
					playing = true;
				}}
				onpause={() => {
					playing = false;
				}}
				onerror={() => {
					videoError = true;
				}}
				class="w-full h-full"
				preload="auto"
			>
				<track kind="captions" />
			</video>

			{#if videoError}
				<div class="absolute inset-0 flex items-center justify-center bg-black/80 rounded-xl">
					<p class="text-sm text-red-400 text-center px-4">
						Could not load video.<br />
						<span class="text-white/50 text-xs">Unsupported format or file not found.</span>
					</p>
				</div>
			{:else if displayWidth > 0 && displayHeight > 0 && annotations}
				<AnnotationCanvas
					{annotations}
					frame={effectiveFrame}
					{toggles}
					width={displayWidth}
					height={displayHeight}
				/>
			{:else if displayWidth > 0 && displayHeight > 0}
				<div class="absolute inset-0 flex items-end p-3 pointer-events-none">
					<span class="text-xs text-white/50">Loading annotations…</span>
				</div>
			{/if}

			<CourtMinimap
				{annotations}
				frame={effectiveFrame}
				visible={toggles.minimap}
				colorMode={toggles.colorMode}
				trails={toggles.speedTrails ? courtTrails : {}}
				bind:minimapCanvas
				bind:dragRight={minimapDragRight}
				bind:dragBottom={minimapDragBottom}
			/>
		</div>

		<!-- Export overlay -->
		{#if exporting}
			<div class="w-full rounded-xl bg-[var(--color-surface)] px-4 py-2">
				<div class="flex items-center gap-3">
					<span class="text-xs font-medium whitespace-nowrap">Exporting… {exportProgress}%</span>
					<div class="flex-1 h-1.5 rounded-full bg-[var(--color-border)] overflow-hidden">
						<div
							class="h-full bg-green-500 rounded-full transition-[width] duration-300"
							style="width: {exportProgress}%"
						></div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Controls bar -->
		<div class="flex items-center gap-4 bg-[var(--color-surface)] rounded-xl px-4 py-3">
			<!-- Play/Pause -->
			<button
				onclick={togglePlay}
				class="text-[var(--color-text)] hover:text-[var(--color-accent)] transition-colors"
				aria-label={playing ? 'Pause' : 'Play'}
			>
				{#if playing}
					<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"
						><rect x="6" y="4" width="4" height="16" /><rect
							x="14"
							y="4"
							width="4"
							height="16"
						/></svg
					>
				{:else}
					<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"
						><polygon points="5,3 19,12 5,21" /></svg
					>
				{/if}
			</button>

			<!-- Time -->
			<span class="text-sm text-[var(--color-text-muted)] whitespace-nowrap min-w-[80px]">
				{formatTime(currentTime)} / {formatTime(displayDuration)}
			</span>

			<!-- Seek + event timeline -->
			<div class="flex-1 min-w-0 self-stretch flex items-center">
				<SeekBarTimeline
					{annotations}
					{currentTime}
					duration={displayDuration}
					onSeek={(t) => {
						video.currentTime = t;
						courtTrails = {};
					}}
				/>
			</div>

			<!-- Frame counter -->
			<span class="text-xs text-[var(--color-text-muted)] whitespace-nowrap">
				F{currentFrame}
			</span>

			<!-- Speed selector -->
			<div class="flex gap-1">
				{#each [0.25, 0.5, 1, 2] as rate}
					<button
						onclick={() => setRate(rate)}
						class="text-xs px-2 py-1 rounded-md transition-colors
						{playbackRate === rate
							? 'bg-[var(--color-accent)] text-white'
							: 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'}"
					>
						{rate}x
					</button>
				{/each}
			</div>

			<!-- Export -->
			<button
				onclick={exportVideo}
				disabled={exporting || !annotations}
				class="text-xs px-3 py-1.5 rounded-md border border-[var(--color-border)] transition-colors
				{exporting
					? 'opacity-50 cursor-not-allowed'
					: 'hover:bg-[var(--color-accent)]/20 hover:text-[var(--color-accent)] hover:border-[var(--color-accent)]/40'}"
			>
				Export
			</button>
		</div>
	</div>
</div>

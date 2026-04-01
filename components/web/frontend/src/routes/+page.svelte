<script lang="ts">
	import { onMount } from 'svelte';
	import { uploadVideo, listJobs, getProgress } from '$lib/api';
	import type { Job } from '$lib/types';

	const VIEW_ONLY = import.meta.env.VITE_VIEW_ONLY === 'true';

	let jobs = $state<Job[]>([]);
	let uploading = $state(false);
	let dragOver = $state(false);
	let error = $state('');
	let pendingFile = $state<File | null>(null);
	let videoName = $state('');
	let trackerType = $state<'flow' | 'hungarian' | 'appearance'>('flow');
	let progressMap = $state<Record<string, { stage: string; pct: number }>>({});
	let pollTimer: ReturnType<typeof setInterval>;

	onMount(() => {
		loadJobs();
		pollTimer = setInterval(tick, 2000);
		return () => clearInterval(pollTimer);
	});

	async function loadJobs() {
		try {
			jobs = await listJobs();
		} catch {}
	}

	async function tick() {
		await loadJobs();
		const processing = jobs.filter((j) => j.status === 'processing');
		const updates: Record<string, { stage: string; pct: number }> = { ...progressMap };
		await Promise.all(
			processing.map(async (j) => {
				try {
					updates[j.id] = await getProgress(j.id);
				} catch {}
			})
		);
		progressMap = updates;
	}

	async function handleUpload(file: File) {
		if (!file.type.startsWith('video/')) {
			error = 'Please upload a video file';
			return;
		}
		// Show name input before uploading.
		pendingFile = file;
		videoName = file.name.replace(/\.[^.]+$/, ''); // pre-fill without extension
	}

	async function confirmUpload() {
		if (!pendingFile) return;
		uploading = true;
		error = '';
		const file = pendingFile;
		pendingFile = null;
		try {
			await uploadVideo(file, videoName, trackerType);
			videoName = '';
			trackerType = 'flow';
			await loadJobs();
		} catch (e: any) {
			error = e.message || 'Upload failed';
		} finally {
			uploading = false;
		}
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const file = e.dataTransfer?.files[0];
		if (file) handleUpload(file);
	}

	function onFileInput(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (file) handleUpload(file);
		input.value = '';
	}

	function statusColor(status: string) {
		if (status === 'done') return 'bg-green-500/20 text-green-400';
		if (status === 'processing') return 'bg-blue-500/20 text-blue-400';
		if (status === 'failed') return 'bg-red-500/20 text-red-400';
		return 'bg-yellow-500/20 text-yellow-400';
	}

	function formatDate(iso: string) {
		return new Date(iso).toLocaleString();
	}
</script>

<svelte:head>
	<title>Basketball Analytics</title>
</svelte:head>

<main class="max-w-6xl mx-auto px-6 py-10">
	{#if !VIEW_ONLY}
		<!-- Drop zone -->
		{#if !pendingFile && !uploading}
			<div
				class="border-2 border-dashed rounded-2xl p-12 text-center transition-colors cursor-pointer
					{dragOver
					? 'border-[var(--color-accent)] bg-[var(--color-accent)]/5'
					: 'border-[var(--color-border)] hover:border-[var(--color-text-muted)]'}"
				role="button"
				tabindex="0"
				ondragover={(e) => {
					e.preventDefault();
					dragOver = true;
				}}
				ondragleave={() => {
					dragOver = false;
				}}
				ondrop={onDrop}
				onclick={() => document.getElementById('file-input')?.click()}
				onkeydown={(e) => {
					if (e.key === 'Enter') document.getElementById('file-input')?.click();
				}}
			>
				<input id="file-input" type="file" accept="video/*" class="hidden" onchange={onFileInput} />
				<div class="text-4xl mb-3">🏀</div>
				<p class="text-lg font-medium text-[var(--color-text)]">
					Drop a video here or click to upload
				</p>
				<p class="text-sm text-[var(--color-text-muted)] mt-2">
					MP4 — up to 10 seconds (for best results)
				</p>
			</div>
		{/if}

		<!-- Name confirmation step -->
		{#if pendingFile}
			<div class="border rounded-2xl p-8 bg-[var(--color-surface)]">
				<p class="text-sm text-[var(--color-text-muted)] mb-1">Selected file</p>
				<p class="font-medium mb-5 truncate">{pendingFile.name}</p>
				<label class="block text-sm font-medium mb-1.5" for="video-name-input">Video name</label>
				<input
					id="video-name-input"
					type="text"
					bind:value={videoName}
					placeholder="Enter a display name…"
					class="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)] mb-5"
					onkeydown={(e) => {
						if (e.key === 'Enter') confirmUpload();
					}}
				/>
				<label class="block text-sm font-medium mb-1.5" for="tracker-type-select">Tracker</label>
				<select
					id="tracker-type-select"
					bind:value={trackerType}
					class="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)] mb-5"
				>
					<option value="flow">FlowTracker (offline, min-cost max-flow)</option>
					<option value="hungarian">HungarianTracker (online, cascaded Hungarian)</option>
					<option value="appearance">AppearanceTracker (online, Hungarian + ReID embeddings)</option
					>
				</select>
				<div class="flex gap-3">
					<button
						onclick={confirmUpload}
						class="flex-1 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white hover:opacity-90 transition-opacity"
					>
						Upload &amp; Process
					</button>
					<button
						onclick={() => {
							pendingFile = null;
							videoName = '';
						}}
						class="rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
					>
						Cancel
					</button>
				</div>
			</div>
		{/if}

		<!-- Uploading spinner -->
		{#if uploading}
			<div class="border-2 border-dashed rounded-2xl p-12 text-center border-[var(--color-border)]">
				<div class="flex items-center justify-center gap-3">
					<div
						class="w-6 h-6 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin"
					></div>
					<span class="text-[var(--color-text-muted)]">Uploading…</span>
				</div>
			</div>
		{/if}

		{#if error}
			<p class="mt-4 text-sm text-red-400">{error}</p>
		{/if}
	{/if}

	<!-- Videos Grid -->
	{#if jobs.length > 0}
		<h2 class="text-xl font-semibold mt-12 mb-6">Videos</h2>
		<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each jobs as job}
				<a
					href={job.status === 'done' ? `/player/${job.id}` : undefined}
					class="block rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5 transition-colors
						{job.status === 'done' ? 'hover:bg-[var(--color-surface-hover)] cursor-pointer' : 'cursor-default'}"
				>
					<div class="flex items-center justify-between mb-3">
						<span class="font-medium truncate mr-3">{job.display_name ?? job.video_name}</span>
						<span
							class="text-xs px-2.5 py-1 rounded-full font-medium whitespace-nowrap {statusColor(
								job.status
							)}"
						>
							{job.status}
						</span>
					</div>
					<p class="text-xs text-[var(--color-text-muted)]">{formatDate(job.created_at)}</p>
					{#if job.error}
						<p class="text-xs text-red-400 mt-2 truncate">{job.error}</p>
					{/if}
					{#if job.status === 'processing'}
						<div class="mt-3 h-1.5 rounded-full bg-[var(--color-border)] overflow-hidden">
							<div
								class="h-full bg-blue-500 rounded-full transition-[width] duration-700 ease-out"
								style="width: {Math.round((progressMap[job.id]?.pct ?? 0) * 100)}%"
							></div>
						</div>
						{#if progressMap[job.id]?.stage}
							<p class="text-xs text-blue-400 mt-1.5 font-mono truncate">
								{progressMap[job.id].stage}
							</p>
						{/if}
					{/if}
				</a>
			{/each}
		</div>
	{/if}
</main>

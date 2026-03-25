<script lang="ts">
	import { onMount } from 'svelte';
	import { uploadVideo, listJobs, getProgress } from '$lib/api';
	import type { Job } from '$lib/types';

	let jobs = $state<Job[]>([]);
	let uploading = $state(false);
	let dragOver = $state(false);
	let error = $state('');
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
		uploading = true;
		error = '';
		try {
			await uploadVideo(file);
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
	<!-- Upload Area -->
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
		{#if uploading}
			<div class="flex items-center justify-center gap-3">
				<div
					class="w-6 h-6 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin"
				></div>
				<span class="text-[var(--color-text-muted)]">Uploading…</span>
			</div>
		{:else}
			<div class="text-4xl mb-3">🏀</div>
			<p class="text-lg font-medium text-[var(--color-text)]">
				Drop a video here or click to upload
			</p>
			<p class="text-sm text-[var(--color-text-muted)] mt-2">MP4, MOV, AVI — up to 2 GB</p>
		{/if}
	</div>

	{#if error}
		<p class="mt-4 text-sm text-red-400">{error}</p>
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
						<span class="font-medium truncate mr-3">{job.video_name}</span>
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

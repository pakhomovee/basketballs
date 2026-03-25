/** Typed API client for the basketball analytics backend. */
import type { AnnotationData, Comment, Job } from './types';

const BASE = '/api';

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
	const res = await fetch(url, init);
	if (!res.ok) {
		const text = await res.text().catch(() => res.statusText);
		throw new Error(`${res.status}: ${text}`);
	}
	return res.json();
}

export async function uploadVideo(file: File): Promise<string> {
	const form = new FormData();
	form.append('file', file);
	const data = await fetchJSON<{ job_id: string }>(`${BASE}/videos`, {
		method: 'POST',
		body: form
	});
	return data.job_id;
}

export async function listJobs(): Promise<Job[]> {
	const data = await fetchJSON<{ jobs: Job[] }>(`${BASE}/videos`);
	return data.jobs;
}

export async function getJob(jobId: string): Promise<Job> {
	return fetchJSON<Job>(`${BASE}/videos/${encodeURIComponent(jobId)}`);
}

export async function getAnnotations(jobId: string): Promise<AnnotationData> {
	return fetchJSON<AnnotationData>(`${BASE}/videos/${encodeURIComponent(jobId)}/annotations`);
}

export function videoUrl(jobId: string): string {
	return `${BASE}/videos/${encodeURIComponent(jobId)}/video`;
}

export async function addComment(
	jobId: string,
	timestampSec: number,
	text: string
): Promise<Comment> {
	return fetchJSON<Comment>(`${BASE}/videos/${encodeURIComponent(jobId)}/comments`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ timestamp_sec: timestampSec, text })
	});
}

export async function getComments(jobId: string): Promise<Comment[]> {
	const data = await fetchJSON<{ comments: Comment[] }>(
		`${BASE}/videos/${encodeURIComponent(jobId)}/comments`
	);
	return data.comments;
}

export async function getProgress(jobId: string): Promise<{ stage: string; pct: number }> {
	return fetchJSON<{ stage: string; pct: number }>(
		`${BASE}/videos/${encodeURIComponent(jobId)}/progress`
	);
}

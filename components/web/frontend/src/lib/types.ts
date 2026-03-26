/** TypeScript interfaces matching the annotation JSON schema. */

export interface VideoMeta {
	fps: number;
	width: number;
	height: number;
	total_frames: number;
	video_name: string;
}

export interface PlayerAnnotation {
	player_id: number;
	team_id: number | null;
	bbox: [number, number, number, number] | null;
	confidence: number | null;
	jersey_number: string | null;
	jersey_confidence: number | null;
	court_position: [number, number] | null;
	speed: number | null;
	skeleton: [number, number, number][] | null;
	mask_polygon: [number, number][] | null;
}

export interface BallAnnotation {
	bbox: [number, number, number, number] | null;
	confidence: number | null;
}

export interface ReidMatrix {
	row_ids: number[];
	col_ids: number[];
	distances: number[][];
}

export interface FrameAnnotation {
	players: PlayerAnnotation[];
	balls: BallAnnotation[];
	reid_cross_frame_matrix: ReidMatrix | null;
}

export interface AnnotationData {
	metadata: VideoMeta;
	frames: Record<string, FrameAnnotation>;
}

export type JobStatus = 'queued' | 'processing' | 'done' | 'failed';

export interface Job {
	id: string;
	status: JobStatus;
	video_name: string;
	display_name: string | null;
	created_at: string;
	error: string | null;
}

export interface Comment {
	id: number;
	job_id: string;
	timestamp_sec: number;
	text: string;
	created_at: string;
}

/** Toggle state for annotation layers. */
export interface ToggleState {
	bboxes: boolean;
	masks: boolean;
	skeletons: boolean;
	ball: boolean;
	jerseyNumbers: boolean;
	playerIds: boolean;
	speedTrails: boolean;
	minimap: boolean;
	reidMatrix: boolean;
	/** Whether to color players by team assignment or by unique player ID. */
	colorMode: 'team' | 'id';
}

export const DEFAULT_TOGGLES: ToggleState = {
	bboxes: true,
	masks: false,
	skeletons: false,
	ball: true,
	jerseyNumbers: false,
	playerIds: true,
	speedTrails: false,
	minimap: true,
	reidMatrix: false,
	colorMode: 'team'
};

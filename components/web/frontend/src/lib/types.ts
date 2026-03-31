/** TypeScript interfaces matching the annotation JSON schema. */

export interface VideoMeta {
	fps: number;
	width: number;
	height: number;
	total_frames: number;
	frame_step?: number;
	video_name: string;
}

export interface PlayerAnnotation {
	player_id: number;
	/** Pre-stitching tracker ID (equals player_id when stitching was not run). */
	raw_player_id: number;
	team_id: number | null;
	bbox: [number, number, number, number] | null;
	confidence: number | null;
	jersey_number: string | null;
	jersey_confidence: number | null;
	court_position: [number, number] | null;
	speed: number | null;
	skeleton: [number, number, number][] | null;
	mask_polygon: [number, number][] | null;
	is_possession: boolean;
	track_number: number | null;
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

export interface PassEventAnnotation {
	frame_start: number;
	frame_end: number;
	from_player_id: number;
	to_player_id: number;
	from_track_number: number | null;
	to_track_number: number | null;
	team_id: number;
	timestamp_sec: number;
}

export interface ShotEventAnnotation {
	frame_start: number;
	frame_end: number;
	is_make: boolean;
	make_start: number | null;
	make_end: number | null;
	timestamp_start_sec: number;
	timestamp_end_sec: number;
	make_timestamp_start_sec: number | null;
	make_timestamp_end_sec: number | null;
	shooter_player_id: number | null;
	shooter_track_number: number | null;
	shooter_team_id: number | null;
}

export interface AnnotationData {
	metadata: VideoMeta;
	frames: Record<string, FrameAnnotation>;
	pass_events: PassEventAnnotation[];
	shot_events?: ShotEventAnnotation[];
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
	trackNumbers: boolean;
	rawJerseyNumbers: boolean;
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
	trackNumbers: true,
	rawJerseyNumbers: false,
	playerIds: true,
	speedTrails: false,
	minimap: true,
	reidMatrix: false,
	colorMode: 'team'
};

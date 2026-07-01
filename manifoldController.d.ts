/**
 * Projects monolith safetensors into granular I7 manifold shards.
 * Uses strict offset arithmetic to prevent BFloat16/Float32 alignment drift.
 * @returns {Promise<void>}
 */
export function projectAllLayers(): Promise<void>;
export const I7_DIR: string;

// Packs h0(k) together with conj(h0(-k)) so the time update needs one fetch.

@group(0) @binding(0) var h0In: texture_2d_array<f32>;
@group(0) @binding(1) var h0Out: texture_storage_2d_array<rgba32float, write>;

const N: u32 = 256u;

@compute @workgroup_size(8, 8, 1)
fn packConjugate(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= N || id.y >= N || id.z >= CASCADES) { return; }
  let h = textureLoad(h0In, id.xy, id.z, 0).xy;
  let mirrored = vec2u((N - id.x) % N, (N - id.y) % N);
  let hm = textureLoad(h0In, mirrored, id.z, 0).xy;
  textureStore(h0Out, id.xy, id.z, vec4f(h, hm.x, -hm.y));
}

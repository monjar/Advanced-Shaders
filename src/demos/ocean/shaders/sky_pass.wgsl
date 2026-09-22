// Fullscreen sky behind everything, plus a far "distance" for the water pass.

@group(0) @binding(0) var<uniform> F: Frame;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) ndc: vec2f,
};

struct FSOut {
  @location(0) color: vec4f,
  @location(1) distance: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  out.position = fullscreenPosition(vi);
  out.ndc = out.position.xy;
  return out;
}

@fragment
fn fs(in: VSOut) -> FSOut {
  let p = F.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let dir = normalize(p.xyz / p.w - F.camPos);
  var out: FSOut;
  out.color = vec4f(skyRadiance(dir, true), 1.0);
  out.distance = 1e6;
  return out;
}

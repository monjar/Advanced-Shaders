// Depth-only pass from the sun for cast shadows.

@group(0) @binding(0) var<uniform> F: Frame;

@vertex
fn vs(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return F.lightViewProj * vec4f(position, 1.0);
}

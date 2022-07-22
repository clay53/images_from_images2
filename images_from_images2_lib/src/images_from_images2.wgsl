struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn source_image_load_vert(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    if (vertex_index == 0u) {
        out.position = vec4<f32>(1.0, -1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, 1.0);
    } else if (vertex_index == 1u) {
        out.position = vec4<f32>(1.0, 3.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, -1.0);
    } else {
        out.position = vec4<f32>(-3.0, -1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(-1.0, 1.0);
    }
    return out;
}

@group(0) @binding(0)
var source_image_load_texture: texture_2d<f32>;

@group(0) @binding(1)
var source_image_load_sampler: sampler;

@fragment
fn source_image_load_frag(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_image_load_texture, source_image_load_sampler, in.tex_coords);
}

@group(0) @binding(0)
var source_images: texture_2d_array<f32>;

@group(0) @binding(1)
var target_image: texture_2d<f32>;

@group(0) @binding(2)
var overlay_output: texture_storage_2d<rgba8unorm, write>;

struct OverlayContext {
    source_image_count: u32,
    source_width: u32,
    source_height: u32,
}

@group(0) @binding(3)
var<uniform> overlay_context: OverlayContext;

@compute @workgroup_size(256)
fn overlay_target(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let overlay_origin = vec2<u32>(gid.x*overlay_context.source_width, gid.y*overlay_context.source_height);

    var min = 0u;
    var min_dist = 99999999999999999999999999999999999999.999999999999999999999999999999999999999999999;
    for(var i = 0u; i < overlay_context.source_image_count; i++) {
        var dist = 0.0;
        for(var x = 0u; x < overlay_context.source_width; x++) {
            for(var y = 0u; y < overlay_context.source_height; y++) {
                dist += distance(textureLoad(target_image, vec2<i32>(vec2<u32>(x, y)+overlay_origin), 0), textureLoad(source_images, vec2<i32>(vec2<u32>(x, y)), i32(i), 0));
            }
        }
        if dist < min_dist {
            min = i;
            min_dist = dist;
        }
    }

    for(var x = 0u; x < overlay_context.source_width; x++) {
        for(var y = 0u; y < overlay_context.source_height; y++) {
            textureStore(overlay_output, vec2<i32>(vec2<u32>(x, y)+overlay_origin), textureLoad(source_images, vec2<i32>(vec2<u32>(x, y)), i32(min), 0));
        }
    }
}
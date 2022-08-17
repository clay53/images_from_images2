struct SourceImageContext {
    rescale: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
}

@group(0) @binding(2)
var<uniform> source_image_context: SourceImageContext;

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
        out.position = vec4<f32>(source_image_context.p1.x, source_image_context.p2.y, 0.0, 1.0);
        out.tex_coords = vec2<f32>(0.0, 1.0)*source_image_context.rescale;
    } else if (vertex_index == 1u) {
        out.position = vec4<f32>(source_image_context.p1, 0.0, 1.0);
        out.tex_coords = vec2<f32>(0.0, 0.0)*source_image_context.rescale;
    } else if (vertex_index == 2u) {
        out.position = vec4<f32>(source_image_context.p2, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, 1.0)*source_image_context.rescale;
    } else {
        out.position = vec4<f32>(source_image_context.p2.x, source_image_context.p1.y, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, 0.0)*source_image_context.rescale;
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
    pack_size: u32,
    pack_width: u32,
    pack_height: u32,
    source_image_count: u32,
    source_width: u32,
    source_height: u32,
}

@group(0) @binding(3)
var<uniform> overlay_context: OverlayContext;

@compute @workgroup_size(1)
fn overlay_target(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let overlay_origin = vec2<u32>(gid.x*overlay_context.source_width, gid.y*overlay_context.source_height);

    var min_pack = 0u;
    var min_offsetx = 0u;
    var min_offsety = 0u;
    var min_dist = 99999999999999999999999999999999999999.999999999999999999999999999999999999999999999;
    for(var i = 0u; i < overlay_context.source_image_count; i++) {
        var pack = i/overlay_context.pack_size;
        var pack_index = i % overlay_context.pack_size;
        var offsetx = (pack_index % overlay_context.pack_width)*overlay_context.source_width;
        var offsety = pack_index/overlay_context.pack_width*overlay_context.source_height;
        var dist = 0.0;
        for(var x = 0u; x < overlay_context.source_width; x++) {
            for(var y = 0u; y < overlay_context.source_height; y++) {
                dist += distance(textureLoad(target_image, vec2<i32>(vec2<u32>(x, y)+overlay_origin), 0), textureLoad(source_images, vec2<i32>(vec2<u32>(x+offsetx, y+offsety)), i32(pack), 0));
            }
        }
        if dist < min_dist {
            min_pack = pack;
            min_offsetx = offsetx;
            min_offsety = offsety;
            min_dist = dist;
        }
    }

    for(var x = 0u; x < overlay_context.source_width; x++) {
        for(var y = 0u; y < overlay_context.source_height; y++) {
            textureStore(overlay_output, vec2<i32>(vec2<u32>(x, y)+overlay_origin), textureLoad(source_images, vec2<i32>(vec2<u32>(min_offsetx+x, min_offsety+y)), i32(min_pack), 0));
        }
    }
}
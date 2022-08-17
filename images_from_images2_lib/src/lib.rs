use std::{path::PathBuf, num::NonZeroU32, time::Instant};

use image::GenericImageView;
use log::{warn, trace, error};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum Error {
    CantFitSourceImages,
    FailedToLoadTargetImage(std::io::Error),
    FailedToDecodeTargetImage(image::ImageError),
}

pub async fn go(source_image_paths: &Vec<PathBuf>, source_width: u32, source_height: u32, target_image_path: &PathBuf) -> Result<ImageOutput, Error> {
    use Error::*;

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, backend, None).await.unwrap();
    
    let requested_source_image_count = source_image_paths.len() as u32;
    let max_texture_array_layers = adapter.limits().max_texture_array_layers;
    let max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;

    let min_pack_size = requested_source_image_count/max_texture_array_layers + u32::from(requested_source_image_count % max_texture_array_layers != 0);
    
    let max_columns = max_texture_dimension_2d/source_width;
    let max_rows = max_texture_dimension_2d/source_height;
    
    let width_prediction = (min_pack_size as f32*source_height as f32/source_width as f32).sqrt();

    let pack_width = if (width_prediction.floor()*source_width as f32*(min_pack_size as f32/width_prediction.floor()).ceil()*source_height as f32) < width_prediction.ceil()*source_width as f32*(min_pack_size as f32/width_prediction.ceil()).ceil()*source_height as f32 && width_prediction.floor() as u32 != 0 {
        width_prediction.floor() as u32
    } else {
        width_prediction.ceil() as u32
    };
    let pack_height = min_pack_size/pack_width + u32::from(min_pack_size % pack_width != 0);

    if pack_width > max_columns || pack_height > max_rows {
        return Err(Error::CantFitSourceImages);
    }

    let pack_size = pack_width*pack_height;
    
    let packs = (requested_source_image_count/pack_size + u32::from(requested_source_image_count % pack_size != 0)).max(2);

    let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
    limits.max_texture_array_layers = packs;
    limits.max_texture_dimension_2d = max_texture_dimension_2d;
    limits.max_storage_textures_per_shader_stage = 1;
    limits.max_compute_workgroup_size_x = 256;
    limits.max_compute_workgroup_size_y = 1;
    limits.max_compute_workgroup_size_z = 1;
    limits.max_compute_workgroups_per_dimension = 65535;
    limits.max_compute_invocations_per_workgroup = 256;
    limits.max_buffer_size = adapter.limits().max_buffer_size;

    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::TEXTURE_BINDING_ARRAY,
            limits,
            label: None,
        },
        None,
    ).await.unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("images_from_images2.wgsl"));

    // load target image
    trace!("Loading target image");
    let target_image = match image::io::Reader::open(target_image_path) {
        Ok(reader) => match reader.decode() {
            Ok(image) => image,
            Err(err) => {
                return Err(FailedToDecodeTargetImage(err))
            }
        },
        Err(err) => {
            return Err(FailedToLoadTargetImage(err))
        }
    };

    let (target_width, target_height) = target_image.dimensions();

    let target_texture_size = wgpu::Extent3d {
        width: target_width,
        height: target_height,
        depth_or_array_layers: 1
    };

    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Target Texture"),
        size: target_texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    queue.write_texture(
        target_texture.as_image_copy(),
        &target_image.into_rgba8(),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(target_width*4),
            rows_per_image: std::num::NonZeroU32::new(target_height)
        },
        target_texture_size
    );

    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // load source images

    let source_textures = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Source Textures"),
        size: wgpu::Extent3d {
            width: source_width*pack_width,
            height: source_height*pack_height,
            depth_or_array_layers: packs,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    });

    let source_image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let source_image_load_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Source Image Load Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }
                },
                count: None, // See if I could make it arrayed
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    });

    let source_image_load_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Source Image Load Pipeline"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Source Image Load Pipeline Layout"),
            bind_group_layouts: &[&source_image_load_bind_group_layout],
            push_constant_ranges: &[]
        })),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "source_image_load_vert",
            buffers: &[]
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "source_image_load_frag",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL
            })]
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::NoUninit)]
    struct SourceImageContext {
        texture_rescale: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
    }

    let source_image_context_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Source Image Context Buffer"),
        size: std::mem::size_of::<SourceImageContext>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let mapped_width = 2.0/pack_width as f32;
    let mapped_height = 2.0/pack_height as f32;

    let mut errored = 0;
    let mut i = 0;
    while i < requested_source_image_count && i-errored < requested_source_image_count {
        trace!("Loading image {} of {}", i-errored+1, requested_source_image_count);
        let source_image_path = &source_image_paths[i as usize];
        let image = match image::io::Reader::open(source_image_path) {
            Ok(reader) => match reader.decode() {
                Ok(image) => image,
                Err(err) => {
                    warn!("Failed to decode image: {}", err);
                    errored += 1;
                    i += 1;
                    continue;
                }
            },
            Err(err) => {
                warn!("Failed to open image: {}", err);
                errored += 1;
                i += 1;
                continue;
            }
        };

        let (width, height) = image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        };

        let source_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Source Image"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });

        queue.write_texture(
            source_texture.as_image_copy(),
            &image.to_rgba8(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4*width),
                rows_per_image: std::num::NonZeroU32::new(height),
            },
            texture_size,
        );

        let source_aspect = width as f32/height as f32;
        let target_aspect = source_width as f32/source_height as f32;

        let texture_rescale = if target_aspect > source_aspect {
            [source_aspect/target_aspect, 1.0]
        } else {
            [1.0, target_aspect/source_aspect]
        };

        let current_target_index = i-errored;
        let current_pack = current_target_index/pack_size;
        let current_pack_index = current_target_index % pack_size;
        let y = current_pack_index/pack_width;
        let x = current_pack_index % pack_width;

        let p1 = [-1.0+x as f32*mapped_width, 1.0-y as f32*mapped_height];
        let p2 = [-1.0+(x+1) as f32*mapped_width, 1.0-(y+1) as f32*mapped_height];

        queue.write_buffer(&source_image_context_buffer, 0, bytemuck::cast_slice(&[SourceImageContext {
            texture_rescale,
            p1,
            p2,
        }]));

        let source_image_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let target_view = source_textures.create_view(&wgpu::TextureViewDescriptor {
            base_array_layer: current_pack,
            array_layer_count: std::num::NonZeroU32::new(1),
            ..Default::default()
        });

        let bind_group_descriptor = wgpu::BindGroupDescriptor {
            label: Some("Source Image Load Bind Group"),
            layout: &source_image_load_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_image_view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&source_image_sampler)
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: source_image_context_buffer.as_entire_binding(),
                },
            ]
        };

        let bind_group = device.create_bind_group(&bind_group_descriptor); // note: must be made before render pass
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Source Image Load Encoder")
        });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Source Image Load Render Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }
                })
            ],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&source_image_load_pipeline);
        
        render_pass.set_bind_group(
            0,
            &bind_group,
            &[]
        );
        render_pass.draw(0..4, 0..1);
        drop(render_pass);
        queue.submit(std::iter::once(encoder.finish()));
        i += 1;
    }

    let source_image_count = i-errored;

    if source_image_count < requested_source_image_count {
        warn!("Did not load all images requested due to too many errors in loading");
    }

    // overlay target image

    trace!("Preparing to overlay");
    let column_count = target_width/source_width;
    let row_count = target_height/source_height;

    let output_width = column_count*source_width;
    let output_height = row_count*source_height;

    let output_texture_size = wgpu::Extent3d {
        width: output_width,
        height: output_height,
        depth_or_array_layers: 1
    };

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size: output_texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
    });

    let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let overlay_context_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Overlay Context Buffer"),
        contents: bytemuck::cast_slice(&[OverlayContext {
            pack_size,
            pack_width,
            pack_height,
            source_image_count,
            source_width,
            source_height,
        }]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let overlay_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Overlay Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ]
    });

    let source_textures_view = source_textures.create_view(&wgpu::TextureViewDescriptor::default());

    let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Overlay Bind Group"),
        layout: &overlay_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&source_textures_view)
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&target_view)
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&output_texture_view)
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: overlay_context_buffer.as_entire_binding()
            }
        ]
    });

    let overlay_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Overlay Pipeline"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Overlay pipeline layout"),
            bind_group_layouts: &[
                &overlay_bind_group_layout
            ],
            push_constant_ranges: &[]
        })),
        module: &shader,
        entry_point: "overlay_target"
    });

    let bytes_per_row = 4*output_width;
    let padded_bytes_per_row = bytes_per_row + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT-bytes_per_row%wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: (padded_bytes_per_row*output_height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("Output Buffer"),
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Overlay command encoder"),
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Overlay Compute Pass")
    });

    compute_pass.set_pipeline(&overlay_pipeline);
    compute_pass.set_bind_group(0, &overlay_bind_group, &[]);
    compute_pass.dispatch_workgroups(column_count, row_count, 1);

    drop(compute_pass);

    encoder.copy_texture_to_buffer(
        output_texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(padded_bytes_per_row),
                rows_per_image: NonZeroU32::new(output_height),
            }
        },
        output_texture_size
    );
    trace!("Overlaying");
    let overlay_timer = Instant::now();
    queue.submit(std::iter::once(encoder.finish()));
    while !device.poll(wgpu::Maintain::Poll) {
        
    }
    println!("Overlay took: {}", overlay_timer.elapsed().as_millis());
    trace!("Overlayed");

    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |r| {
        if let Err(err) = r {
            error!("Failed to map output buffer: {}", err);
        } else {
            trace!("Output buffer mapped!");
        }
    });

    trace!("Mapping output buffer...");
    while !device.poll(wgpu::Maintain::Poll) {
        
    }

    trace!("Collecting output data...");
    let padded_data = buffer_slice.get_mapped_range();
    let output_data: Vec<u8> = padded_data
        .chunks(padded_bytes_per_row as usize)
        .map(|chunk| { &chunk[..bytes_per_row as usize] })
        .flatten()
        .map(|b| *b)
        .collect();
    
    drop(padded_data); // crashes when unmapping otherwise
    
    output_buffer.unmap();

    Ok(ImageOutput {
        data: output_data,
        width: output_width,
        height: output_height,
        color_type: image::ColorType::Rgba8,
    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayContext {
    pub pack_size: u32,
    pub pack_width: u32,
    pub pack_height: u32,
    pub source_image_count: u32,
    pub source_width: u32,
    pub source_height: u32,
}

pub struct ImageOutput {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub color_type: image::ColorType,
}
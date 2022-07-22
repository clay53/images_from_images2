use std::path::PathBuf;

use clap::{Parser};

#[derive(Parser)]
#[clap(author="Clayton Hickey", version="v0.0.0", about="Makes images from images mosaic style")]
struct Args {
    #[clap(value_parser, short, long)]
    source_images: PathBuf,

    #[clap(value_parser, long)]
    source_width: u32,

    #[clap(value_parser, long)]
    source_height: u32,
    
    #[clap(value_parser, short, long)]
    target_image: PathBuf,

    #[clap(value_parser, short, long)]
    output_path: PathBuf,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    
    let source_image_paths: Vec<PathBuf> = args.source_images.read_dir().unwrap().map(|dir_entry| dir_entry.unwrap().path()).collect();

    match futures::executor::block_on(images_from_images2_lib::go(
        &source_image_paths,
        args.source_width,
        args.source_height,
        &args.target_image,
    )) {
        Ok(image_output) => {
            println!("Saving...");
            image::save_buffer(args.output_path, image_output.data.as_slice(), image_output.width, image_output.height, image_output.color_type).unwrap();
            println!("Saved!");
        },
        Err(err) => eprintln!("Failed: {:#?}", err)
    }
}

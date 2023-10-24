fn parse_args() -> Result<(String, String), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!("Usage {} model_path image_path", args[0]).into());
    }
    Ok((args[1].clone(), args[2].clone()))
}

use docai_hf_rs::tasks::DocumentClassifierBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, img_path) = parse_args()?;

    let classification_result = DocumentClassifierBuilder::new()
        // .max_results(3) // set max result
        .build_from_file(model_path)? // create a image classifier using model pat
        .classify(&img_path)?; // do inference and generate results

    // show formatted result message
    println!("\n The top classification result is : \n");
    println!("{}", classification_result[0]);

    Ok(())
}

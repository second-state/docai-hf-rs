use std::fmt;

// Define ClassificationResult
pub struct ClassificationResult {
    pub category_name: String,
    pub score: f32,
    pub index: usize,
}

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Category name: \"{}\"\nScore:         {:.8}\nIndex:         {}",
            self.category_name, self.score, self.index
        )
    }
}

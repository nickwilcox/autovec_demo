use cc;

fn main() {
    cc::Build::new()
        .file("src/mix.c")
        .compile("mix");
}
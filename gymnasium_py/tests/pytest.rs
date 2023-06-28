//! Run all python tests with pytest.
#![cfg(feature = "python")]

#[cfg(test)]
mod pytest {
    #[test]
    fn test_pytest() {
        // Arrange
        let mut command = std::process::Command::new("python3");
        let command = command.arg("-m").arg("pytest");

        // Act
        let output = command.output().unwrap();
        println!("{}", std::str::from_utf8(&output.stdout).unwrap());
        eprintln!("{}", std::str::from_utf8(&output.stderr).unwrap());

        // Assert
        assert!(output.status.success())
    }
}

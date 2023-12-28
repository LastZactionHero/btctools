package main

import (
	"net/http"
)

func main() {
	// Define a file server to serve static files from the "static" directory.
	fs := http.FileServer(http.Dir("static"))

	// Register a handler for the root ("/") URL path to serve the index.html file.
	http.Handle("/", fs)

	// Start the HTTP server on port 8080.
	http.ListenAndServe(":8080", nil)
}

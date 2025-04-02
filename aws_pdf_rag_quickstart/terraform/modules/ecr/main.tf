terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.0.2"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "5.6.2"
    }
  }
}

data "archive_file" "build_context" {
  type        = "zip"
  source_dir  = "${var.build_context}/src"
  output_path = "${var.build_context}/repo.zip"
}

resource "aws_ecr_repository" "aws_ecr_repository" {
  name = var.repo_name
}

resource "docker_image" "docker_image" {
  name = "${replace(var.proxy_endpoint, "https://", "")}/${var.repo_name}:latest"
  build {
    context = var.build_context
    target  = var.build_target
  }
  triggers = {
    build_context_hash = data.archive_file.build_context.output_base64sha256
  }
}

resource "docker_registry_image" "media-handler" {
  name          = docker_image.docker_image.name
  keep_remotely = true
}

output "image_url" {
  value = "${replace(var.proxy_endpoint, "https://", "")}/${var.repo_name}@${docker_registry_image.media-handler.sha256_digest}"
}
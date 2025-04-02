variable "app_name" {
  description = "The name of the application"
  type        = string
}

variable "region_name" {
  description = "The name of the region"
  type        = string
}

variable "environment" {
  description = "The environment name (e.g., dev, staging, prod)"
  type        = string
}

variable "subnets" {
  description = "A list of subnet IDs"
  type        = list(string)
}

variable "security_group_id" {
  description = "The security group ID for the ECS service"
  type        = string
}

variable "docker_image_url" {
  description = "The URL of the Docker image"
  type        = string
}

variable "desired_instance_count" {
  type = number
}

variable "opensearch_url" {
  type = string
}

variable "opensearch_port" {
  type = string
}

variable "opensearch_index" {
  type = string
}

variable "bedrock_endpoint" {
  type = string
}
variable "s3_bucket" {
  type = string
}
variable "env_variables" {
  type = list(map(string))
}

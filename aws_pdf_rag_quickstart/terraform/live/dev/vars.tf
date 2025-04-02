variable "product_name" {
  default = "rag-pdf"
  type    = string
}
variable "aws_profile_name" {
  default = "default"
  type    = string
}
variable "env_suffix" {
  description = "suffix to show env tier (nonprod, prod)"
  type        = string
  default     = "dev"
}
variable "customer" {
  type = string
}
variable "creator" {
  type = string
}
variable "region_name" {
  type = string
}
variable "app_name" {
  type = string
}
variable "build_context" {
}
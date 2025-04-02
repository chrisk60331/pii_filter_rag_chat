variable "bucket_name" {}

resource "aws_s3_bucket" "aws_s3_bucket" {
  bucket        = var.bucket_name
  force_destroy = true
}

output "bucket_name" {
  value = aws_s3_bucket.aws_s3_bucket.id
}
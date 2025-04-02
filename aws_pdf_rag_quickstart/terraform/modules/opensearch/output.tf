output "opensearch_endpoint" {
  value = aws_opensearch_domain.opensearch-domain.endpoint
}

output "aws_opensearch_vpc_endpoint" {
  value = aws_opensearch_vpc_endpoint.opensearch-endpoint.endpoint
}
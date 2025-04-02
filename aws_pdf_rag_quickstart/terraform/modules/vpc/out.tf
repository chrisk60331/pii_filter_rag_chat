output "main_vpc_id" {
  value = aws_vpc.main.id
}

output "main_vpc_subnet_ids" {
  value = [for s in aws_subnet.main : s.id]
}

output "main_vpc_security_group_id" {
  value = aws_security_group.main.id
}

output "bedrock_endpoint" {
  value = aws_vpc_endpoint.bedrock.dns_entry[0].dns_name
}

output "public_subnets" {
  value = [for s in aws_subnet.public : s.id]
}
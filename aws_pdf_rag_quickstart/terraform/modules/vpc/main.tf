resource "aws_vpc" "main" {
  instance_tenancy     = "default"
  cidr_block           = var.vpc_cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "vpc-${var.app_name}"
  }
}

resource "aws_subnet" "main" {
  for_each = var.availability_zones

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 4, each.key)
  availability_zone = each.value
  tags = {
    Name = "subnet-${var.app_name}"
  }
}

resource "aws_subnet" "public" {
  for_each                = var.availability_zones
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 2, each.key)
  availability_zone       = each.value
  map_public_ip_on_launch = true
}

resource "aws_security_group" "main" {
  name   = "security-group-${var.app_name}"
  vpc_id = aws_vpc.main.id
}


resource "aws_vpc_security_group_ingress_rule" "tcp-internal" {
  security_group_id = aws_security_group.main.id

  referenced_security_group_id = aws_security_group.main.id
  from_port                    = 0
  ip_protocol                  = "tcp"
  to_port                      = 65535
}

resource "aws_vpc_security_group_ingress_rule" "tcp-inbound" {
  security_group_id = aws_security_group.main.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 443
  ip_protocol = "tcp"
  to_port     = 443
}

resource "aws_vpc_security_group_egress_rule" "tcp" {
  security_group_id = aws_security_group.main.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 0
  ip_protocol       = "tcp"
  to_port           = 65535
}

resource "aws_vpc_endpoint" "bedrock" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region_name}.bedrock-runtime"
  vpc_endpoint_type = "Interface"

  security_group_ids = [
    aws_security_group.main.id,
  ]

  subnet_ids          = [for s in aws_subnet.main : s.id]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "dkr" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region_name}.ecr.dkr"
  vpc_endpoint_type = "Interface"

  security_group_ids = [
    aws_security_group.main.id,
  ]

  subnet_ids          = [for s in aws_subnet.main : s.id]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "logs" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region_name}.logs"
  vpc_endpoint_type = "Interface"

  security_group_ids = [
    aws_security_group.main.id,
  ]

  subnet_ids          = [for s in aws_subnet.main : s.id]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "ecr" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region_name}.ecr.api"
  vpc_endpoint_type = "Interface"

  security_group_ids = [
    aws_security_group.main.id,
  ]

  subnet_ids          = [for s in aws_subnet.main : s.id]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "monitoring" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region_name}.monitoring"
  vpc_endpoint_type = "Interface"

  security_group_ids = [
    aws_security_group.main.id,
  ]

  subnet_ids          = [for s in aws_subnet.main : s.id]
  private_dns_enabled = true
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "main"
  }
}

resource "aws_eip" "nat" {
  domain = "vpc"
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = [for s in aws_subnet.public : s.id][0]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.public
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
}

resource "aws_route_table_association" "private" {
  for_each       = aws_subnet.main
  subnet_id      = each.value.id
  route_table_id = aws_route_table.private.id
}

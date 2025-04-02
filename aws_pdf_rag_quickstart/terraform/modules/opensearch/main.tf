resource "aws_opensearch_domain" "opensearch-domain" {
  domain_name    = "domain-${var.app_name}"
  engine_version = var.engine_version # Change to your desired version

  cluster_config {
    instance_type            = var.instance_type
    instance_count           = var.instance_count
    dedicated_master_enabled = false
    zone_awareness_enabled   = false
  }

  ebs_options {
    ebs_enabled = true
    volume_type = "gp2"
    volume_size = 10
  }

  advanced_options = {
    "rest.action.multi.allow_explicit_index" = "true"
  }

  vpc_options {
    subnet_ids         = [var.subnet_ids[0]]
    security_group_ids = var.security_group_ids
  }

  node_to_node_encryption {
    enabled = true
  }

  encrypt_at_rest {
    enabled = true
  }
  access_policies = data.aws_iam_policy_document.opensearch-policy.json
}

data "aws_iam_policy_document" "opensearch-policy" {
  statement {
    effect = "Allow"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions   = ["es:*"]
    resources = ["*"]
  }
}


resource "aws_opensearch_vpc_endpoint" "opensearch-endpoint" {
  domain_arn = aws_opensearch_domain.opensearch-domain.arn
  vpc_options {
    security_group_ids = var.security_group_ids
    subnet_ids         = var.subnet_ids
  }
}
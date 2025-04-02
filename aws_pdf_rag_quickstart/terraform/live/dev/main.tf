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
  backend "s3" {
    bucket         = "tf-state"
    key            = "terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "tf-state-lock-table"
    encrypt        = true
    profile        = "saml"
  }
}

provider "aws" {
  region  = var.region_name
  profile = var.aws_profile_name
  default_tags {
    tags = {
      Customer = var.customer
      Creator  = var.creator
    }
  }
}

provider "docker" {
  registry_auth {
    address  = data.aws_ecr_authorization_token.token.proxy_endpoint
    username = data.aws_ecr_authorization_token.token.user_name
    password = data.aws_ecr_authorization_token.token.password
  }
}

data "aws_ecr_authorization_token" "token" {}

locals {
  app_name           = "${var.product_name}-${var.env_suffix}"
  opensearch_index   = "opensearch-index-${local.app_name}"
  availability_zones = { 1 : "${var.region_name}a", 2 : "${var.region_name}b" }
  vpc_cidr_block     = "10.0.0.0/22"
  opensearch_port    = 443
  s3_bucket          = "${local.app_name}-data-bucket"
  embed_llm          = "amazon.titan-embed-text-v2:0"
  chat_llm           = "anthropic.claude-3-sonnet-20240229-v1:0"
  log_level          = "WARNING"
}

module "s3" {
  source      = "../../modules/s3"
  bucket_name = "${var.app_name}-data-${var.env_suffix}"
}

module "ecr-opensearch" {
  source         = "../../modules/ecr"
  proxy_endpoint = data.aws_ecr_authorization_token.token.proxy_endpoint
  repo_name      = "${var.app_name}/opensearch"
  region_name    = var.region_name
  build_context  = var.build_context
  build_target   = "opensearch"
}

module "ecr-api" {
  source         = "../../modules/ecr"
  proxy_endpoint = data.aws_ecr_authorization_token.token.proxy_endpoint
  repo_name      = "${var.app_name}/ecsopensearch"
  region_name    = var.region_name
  build_context  = var.build_context
  build_target   = "ecsopensearch"
}

module "ecr-bedrock" {
  source         = "../../modules/ecr"
  proxy_endpoint = data.aws_ecr_authorization_token.token.proxy_endpoint
  repo_name      = "${var.app_name}/bedrock"
  region_name    = var.region_name
  build_context  = var.build_context
  build_target   = "bedrock"
}

module "lambda_bedrock" {
  source             = "../../modules/lambda"
  image_uri          = replace(module.ecr-bedrock.image_url, "https://", "")
  app_name           = local.app_name
  env_name           = var.env_suffix
  security_group_ids = [module.vpc.main_vpc_security_group_id]
  subnet_ids         = module.vpc.main_vpc_subnet_ids
  env_vars = {
    LOG_LEVEL        = "INFO"
    AOSS_URL         = module.opensearch.opensearch_endpoint
    AOSS_PORT        = local.opensearch_port
    INDEX_NAME       = local.opensearch_index
    BEDROCK_ENDPOINT = "https://bedrock-runtime.${var.region_name}.amazonaws.com"
  }
  function_name = "bedrock"
  policy_actions = [
    "bedrock:*",
    "ec2:CreateNetworkInterface",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeSubnets",
    "ec2:DeleteNetworkInterface",
    "ec2:AssignPrivateIpAddresses",
    "ec2:UnassignPrivateIpAddresses",
    "ec2:DescribeSecurityGroups",
    "ec2:DescribeSubnets",
    "ec2:DescribeVpcs",
  ]
}

module "lambda_opensearch" {
  source = "../../modules/lambda"

  image_uri          = replace(module.ecr-opensearch.image_url, "https://", "")
  app_name           = local.app_name
  env_name           = var.env_suffix
  security_group_ids = [module.vpc.main_vpc_security_group_id]
  subnet_ids         = module.vpc.main_vpc_subnet_ids
  env_vars = {
    LOG_LEVEL        = "INFO"
    AOSS_URL         = module.opensearch.opensearch_endpoint
    AOSS_PORT        = local.opensearch_port
    INDEX_NAME       = local.opensearch_index
    BEDROCK_ENDPOINT = "https://bedrock-runtime.${var.region_name}.amazonaws.com"
    S3_BUCKET        = module.s3.bucket_name
  }
  function_name = "opensearch"
  policy_actions = [
    "bedrock:*",
    "elasticsearch:*",
    "s3:GetObject",
    "ec2:CreateNetworkInterface",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeSubnets",
    "ec2:DeleteNetworkInterface",
    "ec2:AssignPrivateIpAddresses",
    "ec2:UnassignPrivateIpAddresses",
    "ec2:DescribeSecurityGroups",
    "ec2:DescribeSubnets",
    "ec2:DescribeVpcs",
  ]
}

module "opensearch" {
  source             = "../../modules/opensearch"
  app_name           = local.app_name
  engine_version     = "OpenSearch_2.13"
  instance_type      = "t3.small.search"
  instance_count     = 1
  security_group_ids = [module.vpc.main_vpc_security_group_id]
  subnet_ids         = module.vpc.main_vpc_subnet_ids
  vpc_id             = module.vpc.main_vpc_id
}

module "vpc" {
  source             = "../../modules/vpc"
  app_name           = local.app_name
  availability_zones = local.availability_zones
  vpc_cidr_block     = local.vpc_cidr_block
  region_name        = var.region_name
}

module "ecs" {
  source                 = "../../modules/ecs"
  app_name               = local.app_name
  environment            = var.env_suffix
  security_group_id      = module.vpc.main_vpc_security_group_id
  subnets                = module.vpc.main_vpc_subnet_ids
  desired_instance_count = 1
  docker_image_url       = module.ecr-api.image_url
  opensearch_url         = module.opensearch.opensearch_endpoint
  opensearch_index       = local.opensearch_index
  bedrock_endpoint       = "https://bedrock-runtime.${var.region_name}.amazonaws.com"
  region_name            = var.region_name
  opensearch_port        = local.opensearch_port
  s3_bucket              = "${local.app_name}-data-bucket"
  env_variables = [
    {
      name  = "AOSS_URL"
      value = module.opensearch.aws_opensearch_vpc_endpoint
    },
    {
      name  = "AOSS_PORT"
      value = tostring(local.opensearch_port)
    },
    {
      name  = "INDEX_NAME"
      value = local.opensearch_index
    },
    {
      name  = "BEDROCK_ENDPOINT"
      value = "https://${module.vpc.bedrock_endpoint}"
    },
    {
      name  = "S3_BUCKET"
      value = local.s3_bucket
    },
    {
      name  = "LOG_LEVEL"
      value = local.log_level
    },
    {
      name  = "EMBED_MODEL"
      value = local.embed_llm
    },
    {
      name  = "CHAT_MODEL"
      value = local.chat_llm
    }
  ]
}
resource "aws_iam_role" "lambda_role" {
  name = "lambda-role-${var.app_name}-${var.function_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "lambda.amazonaws.com"
        },
        Effect = "Allow",
      }
    ]
  })
}

resource "aws_iam_policy" "lambda_policy" {
  name        = "lambda-policy-${var.app_name}-${var.function_name}"
  description = "Policy that grants access for AWS Lambda"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = var.policy_actions,
        Resource = "*",
        Effect   = "Allow"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attach" {
  policy_arn = aws_iam_policy.lambda_policy.arn
  role       = aws_iam_role.lambda_role.name
}


# Lambda function
resource "aws_lambda_function" "lambda_function" {
  function_name = "lambda-function-${var.function_name}-${var.app_name}"
  role          = aws_iam_role.lambda_role.arn
  environment {
    variables = var.env_vars
  }
  package_type = "Image"
  image_uri    = var.image_uri
  memory_size  = 2048
  timeout      = 45

  vpc_config {
    security_group_ids = var.security_group_ids
    subnet_ids         = var.subnet_ids
  }
}


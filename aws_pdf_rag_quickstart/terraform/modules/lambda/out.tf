output "lambda_role_arn" {
  value = aws_iam_role.lambda_role.arn
}
output "arn" {
  value = aws_lambda_function.lambda_function.arn
}
output "name" {
  value = aws_lambda_function.lambda_function.function_name
}
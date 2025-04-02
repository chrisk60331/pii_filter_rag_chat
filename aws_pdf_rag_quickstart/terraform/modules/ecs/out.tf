output "ecs_cluster_name" {
  value = aws_ecs_cluster.aws_application.name
}

output "ecs_service_name" {
  value = aws_ecs_service.aws_application.name
}


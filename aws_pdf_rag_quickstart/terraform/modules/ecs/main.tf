resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.app_name}-ecs-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy_attachment" "aws_application_bedrock" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
}

resource "aws_iam_role_policy_attachment" "aws_application_s3" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_ecs_cluster" "aws_application" {
  name = "${var.app_name}-cluster"
}

resource "aws_ecs_service" "aws_application" {
  name            = "${var.app_name}-service"
  cluster         = aws_ecs_cluster.aws_application.id
  task_definition = aws_ecs_task_definition.aws_application.arn
  desired_count   = var.desired_instance_count
  launch_type     = "FARGATE"
  network_configuration {
    security_groups  = [var.security_group_id]
    subnets          = var.subnets
    assign_public_ip = false
  }
}

resource "aws_ecs_task_definition" "aws_application" {
  family                   = "${var.app_name}-task"
  network_mode             = "awsvpc"
  cpu                      = 4096
  memory                   = 16384
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_execution_role.arn
  container_definitions = jsonencode([
    {
      name              = var.app_name,
      image             = var.docker_image_url,
      cpu               = 4096,
      memory            = 16384,
      memoryReservation = 12000,
      essential         = true,
      portMappings = [
        {
          containerPort = 80
          hostPort      = 80
        }
      ]
      environment = var.env_variables
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.app_name}"
          "awslogs-region"        = var.region_name
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }
}

resource "aws_cloudwatch_log_group" "aws_cloudwatch_log_group" {
  name = "/ecs/${var.app_name}"
}


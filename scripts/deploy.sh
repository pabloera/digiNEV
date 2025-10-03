#!/bin/bash
# Deployment script for digiNEV Pipeline
# Created: 2025-09-30

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="diginev-pipeline"
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   digiNEV Pipeline Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"
echo ""

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
}

# Function to load environment variables
load_env() {
    echo -e "${YELLOW}Loading environment variables...${NC}"

    if [ -f ".env.$ENVIRONMENT" ]; then
        export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
        echo -e "${GREEN}✓ Loaded .env.$ENVIRONMENT${NC}"
    elif [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
        echo -e "${GREEN}✓ Loaded .env${NC}"
    else
        echo -e "${RED}No environment file found${NC}"
        exit 1
    fi
}

# Function to build Docker images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"

    # Build pipeline image
    docker build -t $PROJECT_NAME:$VERSION -f Dockerfile .
    echo -e "${GREEN}✓ Pipeline image built${NC}"

    # Build dashboard image
    docker build -t $PROJECT_NAME-dashboard:$VERSION -f Dockerfile.dashboard .
    echo -e "${GREEN}✓ Dashboard image built${NC}"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"

    # Run unit tests in Docker
    docker run --rm \
        -e ENVIRONMENT=test \
        $PROJECT_NAME:$VERSION \
        python -m pytest tests/ -v

    echo -e "${GREEN}✓ Tests passed${NC}"
}

# Function to deploy with Docker Compose
deploy_compose() {
    echo -e "${YELLOW}Deploying with Docker Compose...${NC}"

    # Stop existing containers
    docker-compose down

    # Pull latest images (if using registry)
    # docker-compose pull

    # Start services
    docker-compose up -d

    # Wait for services to be healthy
    echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
    sleep 10

    # Check health
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✓ Services are running${NC}"
    else
        echo -e "${RED}Services failed to start${NC}"
        docker-compose logs
        exit 1
    fi
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"

    # Apply Kubernetes manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml

    # Wait for deployment
    kubectl rollout status deployment/$PROJECT_NAME -n diginev

    echo -e "${GREEN}✓ Deployed to Kubernetes${NC}"
}

# Function to run health checks
health_check() {
    echo -e "${YELLOW}Running health checks...${NC}"

    # Check pipeline health
    HEALTH_URL="http://localhost:8000/health"
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Pipeline is healthy${NC}"
    else
        echo -e "${YELLOW}! Pipeline health check failed (may not be exposed)${NC}"
    fi

    # Check dashboard
    DASHBOARD_URL="http://localhost:8501"
    if curl -f $DASHBOARD_URL > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Dashboard is accessible${NC}"
    else
        echo -e "${YELLOW}! Dashboard not accessible${NC}"
    fi
}

# Function to backup existing data
backup_data() {
    echo -e "${YELLOW}Backing up existing data...${NC}"

    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR

    # Backup data directories
    if [ -d "data" ]; then
        tar -czf $BACKUP_DIR/data.tar.gz data/
        echo -e "${GREEN}✓ Data backed up${NC}"
    fi

    if [ -d "output" ]; then
        tar -czf $BACKUP_DIR/output.tar.gz output/
        echo -e "${GREEN}✓ Output backed up${NC}"
    fi
}

# Function to show deployment info
show_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Services:"
    echo -e "  Pipeline: ${GREEN}Running${NC}"
    echo -e "  Dashboard: ${GREEN}http://localhost:8501${NC}"
    echo -e "  Redis: ${GREEN}localhost:6379${NC}"
    echo ""
    echo "Commands:"
    echo "  View logs:    docker-compose logs -f"
    echo "  Stop:         docker-compose down"
    echo "  Restart:      docker-compose restart"
    echo ""
}

# Main deployment flow
main() {
    echo -e "${YELLOW}Starting deployment process...${NC}"
    echo ""

    # Check dependencies
    check_dependencies

    # Load environment
    load_env

    # Backup data
    backup_data

    # Build images
    build_images

    # Run tests (optional)
    if [ "$ENVIRONMENT" != "production" ]; then
        run_tests
    fi

    # Deploy based on environment
    case $ENVIRONMENT in
        local|development)
            deploy_compose
            ;;
        kubernetes|k8s)
            deploy_kubernetes
            ;;
        production)
            deploy_compose
            ;;
        *)
            echo -e "${RED}Unknown environment: $ENVIRONMENT${NC}"
            exit 1
            ;;
    esac

    # Health checks
    health_check

    # Show info
    show_info
}

# Run main function
main
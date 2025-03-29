#!/bin/bash

# Colorful terminal output for a bit of flair
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}=== Developer Environment Setup ===${NC}"


# Get user inputs with validation
get_email() {
    local email_regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    local email=""
    
    while [[ ! $email =~ $email_regex ]]; do
        read -p "$(echo -e "${BOLD}Enter your email:${NC} ")" email
        
        if [[ ! $email =~ $email_regex ]]; then
            echo -e "${RED}Invalid email format. Please try again.${NC}"
        fi
    done
    
    echo "$email"
}

get_name() {
    local name=""
    
    while [[ -z $name ]]; do
        read -p "$(echo -e "${BOLD}Enter your name:${NC} ")" name
        
        if [[ -z $name ]]; then
            echo -e "${RED}Name cannot be empty. Please try again.${NC}"
        fi
    done
    
    echo "$name"
}

# Get the user's email and name
USER_EMAIL=$(get_email)
USER_NAME=$(get_name)

echo -e "\n${BOLD}${GREEN}Setting up Git configuration...${NC}"
git config --global user.email "$USER_EMAIL"
git config --global user.name "$USER_NAME"

echo -e "\n${BOLD}${GREEN}Configuring system packages...${NC}"
echo -e "${YELLOW}Updating package lists...${NC}"
sudo apt update

echo -e "\n${YELLOW}Installing Python 3.12...${NC}"
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12
sudo apt-get install -y python3.12-venv python3.12-dev

echo -e "\n${YELLOW}Installing system utilities...${NC}"
sudo apt-get install -y htop

echo -e "\n${BOLD}${GREEN}Setup complete!${NC}"
echo -e "Git configured for: ${BLUE}$USER_NAME${NC} <${BLUE}$USER_EMAIL${NC}>"
echo -e "Python 3.12 and development tools installed"
echo -e "System monitoring tools (htop) installed\n"

echo -e "${BOLD}${GREEN}Happy coding! 🚀${NC}"
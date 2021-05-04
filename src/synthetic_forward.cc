#include <ultra.h>
#include <iostream>

static const char* USAGE = "Usage:\n\t<executabel> <project_workspace_dir>.\n";

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    
    if (argc != 2) {
        std::cout << USAGE << std::endl;
        return 1;
    }
    ultra::forward(argv[1]);
    return 0;
}

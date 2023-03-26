#include <iostream>
#include <flags/flags.h>
#include "util.h"
#include "cppraisr.h"

void print_help()
{
}

int main(int argc, char** argv)
{
    using namespace cppraisr;
    const flags::args args(argc, argv);
    if(args.get<bool>("help", false)){
        print_help();
        return 0;
    }

    std::vector<std::filesystem::path> files = parse_directory("train_data",
                                                               [](const std::filesystem::directory_entry& entry) {
                                                                   std::filesystem::path extension = entry.path().extension();
                                                                   if(extension == ".jpg") {
                                                                       return true;
                                                                   }
                                                                   if(extension == ".png") {
                                                                       return true;
                                                                   }
                                                                   return false;
                                                               });
    std::unique_ptr<RAISRTrainer> trainer = std::make_unique<RAISRTrainer>();
    trainer->train(files, 1);
    return 0;
}

$(document).ready(function() {

        // JQuery code to be added in here.
        // $('.overlay').on('click', function () {
        //         // hide sidebar
        //         $('#sidebar').removeClass('active');
        //         // hide overlay
        //         $('.overlay').removeClass('active');
        //     });
        //
        //     $('#sidebarCollapse').on('click', function () {
        //
        //         // open sidebar
        //         $('#sidebar').addClass('active');
        //         // fade in the overlay
        //         $('.overlay').addClass('active');
        //         $('.collapse.in').toggleClass('in');
        //     });

    $('.file-upload').change(function() {
      // alert("hello");
          $('#form-upload').submit();
        $('#file-label').fadeOut("slow", "linear");
        // $('#form-upload').hide();
        });
});

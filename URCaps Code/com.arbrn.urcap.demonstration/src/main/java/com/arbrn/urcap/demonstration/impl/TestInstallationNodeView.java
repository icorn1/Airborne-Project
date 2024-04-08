package com.arbrn.urcap.demonstration.impl;

import com.ur.urcap.api.contribution.installation.swing.SwingInstallationNodeView;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import java.net.URL;


public class TestInstallationNodeView implements SwingInstallationNodeView<TestInstallationNodeContribution> {

    private Option ACTION_TYPE = Option.Run;
    private Laminate SELECTED_LAMINATE = Laminate.Mountains;

    private enum Option {
        Run("move_ply(", "Run"),
        Calibrate("calibration(", "Calibrate");

        private final String id;
        private final String displayName;

        Option(String id, String displayName) {
            this.id = id;
            this.displayName = displayName;
        }

        public String getId() {
            return id;
        }

        public String getDisplayName() {
            return displayName;
        }
    }

    private enum Laminate {
        Mountains("1", "Mountains", "/com/arbrn/urcap/demonstration/Moun.png"),
        Angrenost("2", "Angrenost", "/com/arbrn/urcap/demonstration/Angr.png"),
        Picasso("3", "Picasso", "/com/arbrn/urcap/demonstration/Pic.png");

        private final String id;
        private final String displayName;
        private final String imagePath;

        Laminate(String id, String displayName, String imagePath) {
            this.id = id;
            this.displayName = displayName;
            this.imagePath = imagePath;
        }

        public String getId() {
            return id;
        }

        public String getDisplayName() {
            return displayName;
        }

        public String getImagePath() {
            return imagePath;
        }
    }

    @Override
    public void buildUI(JPanel panel, TestInstallationNodeContribution contribution) {
        // Main function, where the UI is built & shaped.
        
        // panel is our main panel for the application.
        panel.setLayout(new BorderLayout());

        // Top logos container (includes teh two logos.)
        JPanel topLogoContainer = new JPanel(new FlowLayout(FlowLayout.LEADING, 0, 0)); // Horizontal alignment, no horizontal and vertical gap
        topLogoContainer.add(createThuasLogoPanel());
        topLogoContainer.add(Box.createHorizontalStrut(500));
        topLogoContainer.add(createLogoPanel());
        panel.add(topLogoContainer, BorderLayout.NORTH);

        panel.add(createSpacer(400)); // Add spacing (doesnt work)

        // Settings panel, below the button, bottom left
        JPanel settingsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT)); // Use FlowLayout to left-align components
        JLabel actionText = new JLabel("Action > ");
        JLabel laminateText = new JLabel("Laminate > ");

        settingsPanel.add(actionText, BorderLayout.WEST);
        settingsPanel.add(createActionToPerform());
        settingsPanel.add(laminateText, BorderLayout.EAST);
        settingsPanel.add(createLaminateSelection(panel, contribution));

        panel.add(settingsPanel, BorderLayout.SOUTH);

        // Laminate image panel + start button (same panel)
        addLaminateImagePanel(panel, contribution);
    }

    private JPanel createThuasLogoPanel() {
        JPanel thuasLogoPanel = new JPanel(); // No layout manager specified for automatic sizing
        JLabel thuasLogoLabel = new JLabel();
        ImageIcon thuasLogoIcon = new ImageIcon(getClass().getResource("/com/arbrn/urcap/demonstration/thuaslogo.png"));
        int width = 250;
        int height = (int) ((double) thuasLogoIcon.getIconHeight() / thuasLogoIcon.getIconWidth() * width); // Calculate height to maintain aspect ratio
        thuasLogoLabel.setIcon(new ImageIcon(thuasLogoIcon.getImage().getScaledInstance(width, height, Image.SCALE_SMOOTH)));
        thuasLogoPanel.add(thuasLogoLabel); 
        return thuasLogoPanel;
    }

    private JPanel createLogoPanel() {
        JPanel logoPanel = new JPanel();
        JLabel logoLabel = new JLabel();
        ImageIcon logoIcon = new ImageIcon(getClass().getResource("/com/arbrn/urcap/demonstration/logo.jpg"));
        int width = 250; 
        int height = (int) ((double) logoIcon.getIconHeight() / logoIcon.getIconWidth() * width); // Calculate height to maintain aspect ratio
        logoLabel.setIcon(new ImageIcon(logoIcon.getImage().getScaledInstance(width, height, Image.SCALE_SMOOTH)));
        logoPanel.add(logoLabel); 
        return logoPanel;
    }

    private Box createSenderTestButton(final TestInstallationNodeContribution contribution) {
        // Creates the big green button and makes it functional
        Box box = Box.createVerticalBox();

        JLabel labelText = new JLabel("Press the following button to start the program once you have the desired configuration.");
        labelText.setAlignmentX(Component.CENTER_ALIGNMENT);
        labelText.setFont(new Font("Arial", Font.PLAIN, 20)); // Increase font size
        box.add(labelText);

        final JButton startButton = new JButton("START");
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Here we send to the contribution the options for the script
                contribution.sendScriptTest(ACTION_TYPE.getId(), SELECTED_LAMINATE.getId());
                startButton.setEnabled(false);
                
                try {
                    Thread.sleep(4000);
                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }
                startButton.setEnabled(true);
            }
        });
        startButton.setBackground(new Color(24, 196, 12));
        startButton.setForeground(Color.WHITE);
        Dimension buttonSize = new Dimension(250, 70); 
        startButton.setMaximumSize(buttonSize);
        startButton.setMinimumSize(buttonSize);
        startButton.setPreferredSize(buttonSize);
        startButton.setAlignmentX(Component.CENTER_ALIGNMENT); // Center the button horizontally
        startButton.setFont(new Font("Arial", Font.BOLD, 30)); // Increase button font size
        box.add(createSpacer(20)); // Add space between text and button
        box.add(startButton);

        return box;
    }

    private JComboBox<Option> createActionToPerform() {
        // Creates the preselected text boxes for the Action to perform
        final JComboBox<Option> jcb = new JComboBox<Option>(new DefaultComboBoxModel<Option>(Option.values()));
        jcb.setSelectedIndex(0);
        jcb.setPreferredSize(new Dimension(150, 30));
        jcb.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED) {
                    Option selectedOption = (Option) jcb.getSelectedItem();
                    System.out.println("Selected option: " + selectedOption.getDisplayName());
                    ACTION_TYPE = selectedOption;
                }
            }
        });
        return jcb;
    }



    private JPanel updateSupportImage(String imagePath) {
        // In this function we "update" the laminate image (we just remove it previously and add it now)
        ImageIcon logoIcon = new ImageIcon(getClass().getResource(imagePath));
        if (logoIcon != null) {
            int width = 300; 
            int height = (int) ((double) logoIcon.getIconHeight() / logoIcon.getIconWidth() * width); // Calculate height to maintain aspect ratio
            Image scaledImage = logoIcon.getImage().getScaledInstance(width, height, Image.SCALE_SMOOTH);
            JLabel logoLabel = new JLabel(new ImageIcon(scaledImage));
            JPanel logoPanel = new JPanel(new BorderLayout());
            logoPanel.add(logoLabel, BorderLayout.SOUTH); 

            return logoPanel;
        } else {
            System.out.println("Image not found: " + imagePath);
            return null;
        }
    }

    private JComboBox<Laminate> createLaminateSelection(final JPanel panel, final TestInstallationNodeContribution contribution) {
        // This function returns a Java ComboBox, aka a preselected selection box, with premade laminates.
        final JComboBox<Laminate> jcb = new JComboBox<Laminate>(Laminate.values());
        jcb.setSelectedItem(SELECTED_LAMINATE);
        jcb.setPreferredSize(new Dimension(150, 30));
        jcb.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED) {
                    // Remove the previous image
                    removeLaminateImagePanel(panel);
                    SELECTED_LAMINATE = (Laminate) jcb.getSelectedItem();
                    System.out.println("Selected laminate: " + SELECTED_LAMINATE.getDisplayName());
                    // Add the image
                    addLaminateImagePanel(panel, contribution);
                }
            }
        });
        return jcb;
    }

    private void addLaminateImagePanel(JPanel panel, final TestInstallationNodeContribution contribution){
        // In this function, we add three panels (two here and one in updateSupportImage).
        // The innermost panel contains the laminate image.
        JPanel outerPanel = new JPanel(new BorderLayout());
        JPanel southPanel = new JPanel(new BorderLayout());
        southPanel.add(updateSupportImage(SELECTED_LAMINATE.getImagePath()), BorderLayout.EAST);
        outerPanel.add(southPanel, BorderLayout.SOUTH);
        outerPanel.setName("outerLamPanel");
        outerPanel.add(createSenderTestButton(contribution), BorderLayout.CENTER);
        panel.add(outerPanel, BorderLayout.CENTER); // Add updated panel to the main panel


        panel.revalidate(); // Revalidate the panel to reflect changes
        panel.repaint(); // Repaint the panel to reflect changes
    }

    private void removeLaminateImagePanel(JPanel panel) {
        // In order to update the images accordingly, you need to delete every panel that was added when the laminate
        // image was added, in this case it's three nested panels (although you can just delete the root panel).
        // That's what we do here: find the outermost panel and delete it. It is important to set a name when adding
        // the panel in order to find it now.

        Component[] components = panel.getComponents();
        for (Component component : components) {
            if (component instanceof JPanel) {
                JPanel outerPanel = (JPanel) component;
                if(outerPanel.getName() != null && outerPanel.getName().equals("outerLamPanel")){
                    System.out.println("Image Panel found -> Deleting it");
                    panel.remove(outerPanel);
                    break;
                }
            }
        }
    }

    private Component createSpacer(int height) {
        return Box.createRigidArea(new Dimension(0, height));
    }
}
